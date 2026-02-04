import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb

# RDKit imports for molecular graph creation
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdchem
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.conv import MessagePassing, TransformerConv
from torch_geometric.utils import softmax
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch.nn.init import kaiming_uniform_, zeros_

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class MultiHeadTripletAttention(MessagePassing):
    def __init__(self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs):
        super(MultiHeadTripletAttention, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.heads = heads
        self.negative_slope = negative_slope
        
        # Ensure dimensions are compatible
        self.head_dim = node_channels // heads
        assert node_channels % heads == 0, f"node_channels ({node_channels}) must be divisible by heads ({heads})"
        
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * self.head_dim))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * self.head_dim))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * self.head_dim))
        self.weight_scale = Parameter(torch.Tensor(heads * self.head_dim, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        # Project node and edge features
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        
        # Ensure edge_attr has correct dimensions
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)

        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.head_dim)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias
        return aggr_out

class GraphTransformerBlock(torch.nn.Module):
    def __init__(self, dim, edge_dim, heads=4):
        super(GraphTransformerBlock, self).__init__()
        self.conv = MultiHeadTripletAttention(dim, edge_dim, heads)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        m = F.celu(self.conv.forward(x, edge_index, edge_attr))
        x = self.ln(m)
        return x

class GraphTransformerEncoder(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=300, depth=3, heads=4, dropout=0.1):
        super(GraphTransformerEncoder, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        
        # Ensure hidden_dim is divisible by heads
        assert hidden_dim % heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        
        # Initial node embedding layers - build up to hidden_dim gradually
        self.conv1 = TransformerConv(node_dim, hidden_dim//4, heads=heads//2, dropout=dropout)
        self.conv2 = TransformerConv(hidden_dim//4, hidden_dim//2, heads=heads//2, dropout=dropout)
        self.conv3 = TransformerConv(hidden_dim//2, hidden_dim, heads=heads, dropout=dropout)
        
        # Custom attention block
        self.attention_block = GraphTransformerBlock(hidden_dim, edge_dim, heads)
        
        # Global pooling
        self.global_pool = global_max_pool
        
        # Optional: Add a final projection layer to match your BERT output size (113)
        self.projection = nn.Linear(hidden_dim, 113)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Forward through transformer layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply custom attention
        x = x + F.dropout(self.attention_block(x, edge_index, edge_attr), p=self.dropout, training=self.training)
        
        # Global pooling to get molecular-level representation
        x = self.global_pool(x, batch)
        
        # Project to desired output size
        x = self.projection(x)
        
        return x

def smiles_to_graph(smiles_string):
    """Convert SMILES string to PyTorch Geometric graph data"""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    
    # Add hydrogens for more complete representation
    mol = Chem.AddHs(mol)
    
    # Node features (atoms)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            atom.GetMass(),
        ]
        atom_features.append(features)
    
    # Edge features (bonds)
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add both directions for undirected graph
        edge_indices.extend([[i, j], [j, i]])
        
        bond_features = [
            int(bond.GetBondType()),
            int(bond.GetIsAromatic()),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated()),
        ]
        
        # Add same features for both directions
        edge_features.extend([bond_features, bond_features])
    
    # Convert to tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

class GraphTransformerFeatureExtractor:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model with appropriate dimensions
        # These dimensions are based on the features we extract from molecules
        node_dim = 8  # Number of atom features we extract
        edge_dim = 4  # Number of bond features we extract
        hidden_dim = 300  # Must be divisible by heads
        heads = 4
        
        # Ensure dimensions work properly
        assert hidden_dim % heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        
        self.model = GraphTransformerEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            depth=3,
            heads=heads,
            dropout=0.1
        ).to(self.device)
        
        # If you have a pre-trained model, load it here
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path))
                print(f"Loaded pre-trained model from {model_path}")
            except:
                print("Could not load pre-trained model, using random initialization")
        
        self.model.eval()
    
    def extract_features(self, smiles_list):
        """Extract molecular features from SMILES strings using graph transformer"""
        import psutil
        import time
        
        features = []
        failed_molecules = 0
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        print(f"Extracting graph features for {len(smiles_list)} molecules...")
        print(f"Starting memory usage: {start_memory:.1f}%")
        
        with torch.no_grad():
            for i, smiles in enumerate(smiles_list):
                if i % 100 == 0:
                    print(f"Processing molecule {i+1}/{len(smiles_list)}")
                
                try:
                    # Convert SMILES to graph
                    graph_data = smiles_to_graph(smiles)
                    
                    if graph_data is None:
                        # If molecule parsing fails, create zero vector
                        features.append(np.zeros(113))
                        failed_molecules += 1
                        continue
                    
                    # Add batch dimension and move to device
                    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
                    graph_data = graph_data.to(self.device)
                    
                    # Extract features
                    molecular_features = self.model(graph_data)
                    
                    # Convert to numpy
                    features_np = molecular_features.cpu().numpy().flatten()
                    features.append(features_np)
                    
                except Exception as e:
                    print(f"Error processing molecule {i}: {str(e)}")
                    features.append(np.zeros(113))
                    failed_molecules += 1
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().percent
        
        print(f"Failed to process {failed_molecules} molecules out of {len(smiles_list)}")
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        print(f"Memory usage change: {end_memory - start_memory:.1f}%")
        print(f"Average time per molecule: {(end_time - start_time)/len(smiles_list):.3f} seconds")
        
        return np.array(features)

def find_optimal_threshold(y_true, y_probs):
    """Find optimal threshold for binary classification using F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

def evaluate_model(model, X_test, y_test, threshold=0.5, model_type='sklearn'):
    """Comprehensive model evaluation"""
    if model_type == 'xgboost':
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_probs >= threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Threshold: {threshold:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, roc_auc, f1

def main():
    # Load your existing data
    train_df = pd.read_csv("train_bbbp.csv")
    valid_df = pd.read_csv("valid_bbbp.csv")
    test_df = pd.read_csv("test_bbbp.csv")
    
    # Get SMILES from the 'ids' column (same as your BERT code)
    train_smiles = train_df['ids'].tolist()
    valid_smiles = valid_df['ids'].tolist()
    test_smiles = test_df['ids'].tolist()

    # Initialize graph transformer feature extractor
    feature_extractor = GraphTransformerFeatureExtractor()
    
    # Extract features using graph transformer
    print("Extracting features with Graph Transformer...")
    X_train = feature_extractor.extract_features(train_smiles)
    X_valid = feature_extractor.extract_features(valid_smiles)
    X_test = feature_extractor.extract_features(test_smiles)
    
    # Get labels
    y_train = train_df["y"].values
    y_valid = valid_df["y"].values
    
    # Analyze class distribution
    print("Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_label, count in zip(unique, counts):
        print(f"Class {class_label}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("GRAPH TRANSFORMER FEATURE EVALUATION")
    print("="*60)
    
    # Test the same approaches as your BERT code
    approaches = {}
    
    # 1. Weighted Logistic Regression
    print("\n1. Weighted Logistic Regression:")
    model_weighted = LogisticRegression(
        max_iter=1000, 
        solver='lbfgs', 
        random_state=42,
        class_weight='balanced'
    )
    model_weighted.fit(X_train_scaled, y_train)
    approaches['Weighted LR'] = model_weighted
    evaluate_model(model_weighted, X_valid_scaled, y_valid)
    
    # 2. SMOTE + Logistic Regression
    print("\n2. SMOTE + Logistic Regression:")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    model_smote = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model_smote.fit(X_train_smote, y_train_smote)
    approaches['SMOTE LR'] = model_smote
    evaluate_model(model_smote, X_valid_scaled, y_valid)
    
    # 3. Random Forest
    print("\n3. Random Forest with Class Weights:")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    approaches['Random Forest'] = rf_model
    evaluate_model(rf_model, X_valid_scaled, y_valid)
    
    # 4. XGBoost
    print("\n4. XGBoost with Class Weights:")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train_scaled, y_train)
    approaches['XGBoost'] = xgb_model
    evaluate_model(xgb_model, X_valid_scaled, y_valid, model_type='xgboost')
    
    # Find best model
    best_f1 = 0
    best_model_name = None
    best_model = None
    
    for name, model in approaches.items():
        model_type = 'xgboost' if name == 'XGBoost' else 'sklearn'
        
        if model_type == 'xgboost':
            y_probs = model.predict_proba(X_valid_scaled)[:, 1]
        else:
            y_probs = model.predict_proba(X_valid_scaled)[:, 1]
            
        optimal_threshold = find_optimal_threshold(y_valid, y_probs)
        _, _, f1 = evaluate_model(model, X_valid_scaled, y_valid, optimal_threshold, model_type)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
    
    print(f"\nBest model: {best_model_name} with F1-score: {best_f1:.3f}")
    
    # Generate final test predictions
    model_type = 'xgboost' if best_model_name == 'XGBoost' else 'sklearn'
    
    if model_type == 'xgboost':
        y_test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
        y_valid_probs = best_model.predict_proba(X_valid_scaled)[:, 1]
    else:
        y_test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
        y_valid_probs = best_model.predict_proba(X_valid_scaled)[:, 1]
    
    optimal_threshold = find_optimal_threshold(y_valid, y_valid_probs)
    y_test_pred = (y_test_probs >= optimal_threshold).astype(int)
    
    # Save results
    output_df = pd.DataFrame({
        "ids": test_df["ids"],
        "predicted_label": y_test_pred,
        "predicted_prob": y_test_probs
    })
    
    output_df.to_csv("test_predictions_graph_transformer.csv", index=False)
    print(f"\nTest predictions saved to test_predictions_graph_transformer.csv")
    print(f"Used model: {best_model_name}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")

if __name__ == "__main__":
    main()