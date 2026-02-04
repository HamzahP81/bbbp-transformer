import pandas as pd
import numpy as np
import torch
import os
import sys
import codecs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from transformers import BertTokenizer
from accelerate import Accelerator
import xgboost as xgb

# Import the model
from BERT import BERT, SMILESLM
from SmilesPE.tokenizer import SPE_Tokenizer

class SMILESToBERTFeatureExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seq_length = 100
        
        # Model configuration (same as in the paper)
        self.d_model = 512
        self.n_layers = 4
        self.heads = 8
        self.dropout = 0.1
        
        # Load tokenizers
        self.smiles_tokenizer = BertTokenizer("../data/spe_tokenizer/vocab_spe.txt")
        spe_vob = codecs.open('../data/spe_tokenizer/SPE_ChEMBL.txt')
        self.spe = SPE_Tokenizer(spe_vob)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
    def _load_model(self):
        """Load the pre-trained SMILES-to-BERT model"""
        accelerator = Accelerator()
        
        # Initialize model
        vocab_size = len(self.smiles_tokenizer.vocab)
        bert_model = BERT(vocab_size=vocab_size, d_model=self.d_model, 
                         n_layers=self.n_layers, heads=self.heads,
                         dropout=self.dropout, seq_len=self.seq_length, 
                         device=self.device)
        
        smiles_model = SMILESLM(bert_model=bert_model, output=113)
        smiles_model.to(self.device)
        smiles_model = accelerator.prepare(smiles_model)
        
        # Load checkpoint
        accelerator.load_state(input_dir="../ckpts/checkpoint_19/")
        
        return smiles_model
    
    def _tokenize_smiles(self, smiles):
        """Convert SMILES string to token sequence"""
        try:
            # Use SPE tokenization
            spe_tokens = self.spe.tokenize(smiles).split(' ')
            tokens = self.smiles_tokenizer.encode(spe_tokens)[:-1]  # Remove [SEP]
            
            # Truncate if too long
            if len(tokens) > self.seq_length:
                tokens = tokens[:self.seq_length]
            
            # Pad if too short
            pad_token = self.smiles_tokenizer.encode('[PAD]')[1]
            padding = [pad_token] * (self.seq_length - len(tokens))
            tokens.extend(padding)
            
            return torch.tensor(tokens, dtype=torch.long)
        except:
            # Return padded sequence if tokenization fails
            pad_token = self.smiles_tokenizer.encode('[PAD]')[1]
            return torch.tensor([pad_token] * self.seq_length, dtype=torch.long)
    
    def extract_features(self, smiles_list):
        """Extract 113 molecular descriptors from SMILES strings"""
        features = []
        
        print(f"Extracting features for {len(smiles_list)} molecules...")
        
        with torch.no_grad():
            for i, smiles in enumerate(smiles_list):
                if i % 100 == 0:
                    print(f"Processing molecule {i+1}/{len(smiles_list)}")
                
                # Tokenize and predict
                tokens = self._tokenize_smiles(smiles).unsqueeze(0).to(self.device)
                descriptors, _ = self.model(tokens)
                
                # Convert to numpy
                descriptors_np = descriptors.cpu().numpy().flatten()
                features.append(descriptors_np)
        
        return np.array(features)

def find_optimal_threshold(y_true, y_probs):
    """Find optimal threshold for binary classification using F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    # Handle division by zero
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

def evaluate_model(model, X_test, y_test, threshold=0.5, model_type='sklearn'):
    """Comprehensive model evaluation"""
    if model_type == 'xgboost':
        # For XGBoost, use predict_proba method
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        # For sklearn models
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
    train_df = pd.read_csv("../train_bbbp.csv")
    valid_df = pd.read_csv("../valid_bbbp.csv")
    test_df = pd.read_csv("../test_bbbp.csv")
    
    # put SMILES in lists
    train_smiles = train_df['ids'].tolist()
    valid_smiles = valid_df['ids'].tolist()
    test_smiles = test_df['ids'].tolist()

    # Start feature extractor
    feature_extractor = SMILESToBERTFeatureExtractor()
    
    # Extract features
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
    
    print("\n" + "="*50)
    print("APPROACH 1: CLASS WEIGHTS")
    print("="*50)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    # Train with class weights
    model_weighted = LogisticRegression(
        max_iter=1000, 
        solver='lbfgs', 
        random_state=42,
        class_weight='balanced'
    )
    model_weighted.fit(X_train_scaled, y_train)
    
    print("\nWeighted Logistic Regression Results:")
    evaluate_model(model_weighted, X_valid_scaled, y_valid)
    
    print("\n" + "="*50)
    print("APPROACH 2: SMOTE OVERSAMPLING")
    print("="*50)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Original training set size: {len(y_train)}")
    print(f"SMOTE training set size: {len(y_train_smote)}")
    
    # Train on SMOTE data
    model_smote = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model_smote.fit(X_train_smote, y_train_smote)
    
    print("\nSMOTE Logistic Regression Results:")
    evaluate_model(model_smote, X_valid_scaled, y_valid)
    
    print("\n" + "="*50)
    print("APPROACH 3: ADASYN OVERSAMPLING")
    print("="*50)
    
    # Apply ADASYN
    adasyn = ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_scaled, y_train)
    
    print(f"ADASYN training set size: {len(y_train_adasyn)}")
    
    # Train on ADASYN data
    model_adasyn = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model_adasyn.fit(X_train_adasyn, y_train_adasyn)
    
    print("\nADASYN Logistic Regression Results:")
    evaluate_model(model_adasyn, X_valid_scaled, y_valid)
    
    print("\n" + "="*50)
    print("APPROACH 4: SMOTEENN (COMBINATION)")
    print("="*50)
    
    # Apply SMOTEENN
    smoteenn = SMOTEENN(random_state=42)
    X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train_scaled, y_train)
    
    print(f"SMOTEENN training set size: {len(y_train_smoteenn)}")
    
    # Train on SMOTEENN data
    model_smoteenn = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model_smoteenn.fit(X_train_smoteenn, y_train_smoteenn)
    
    print("\nSMOTEENN Logistic Regression Results:")
    evaluate_model(model_smoteenn, X_valid_scaled, y_valid)
    
    print("\n" + "="*50)
    print("APPROACH 5: RANDOM FOREST WITH CLASS WEIGHTS")
    print("="*50)
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    print("\nRandom Forest Results:")
    evaluate_model(rf_model, X_valid_scaled, y_valid)
    
    print("\n" + "="*50)
    print("APPROACH 6: THRESHOLD OPTIMISATION")
    print("="*50)

    print("\n" + "="*50)
    print("APPROACH 6: XGBOOST WITH CLASS WEIGHTS")
    print("="*50)
    
    # Calculate scale_pos_weight for XGBoost
    # This is equivalent to class weighting for binary classification
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale_pos_weight for XGBoost: {pos_weight:.3f}")
    
    # XGBoost with class balancing
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
    
    print("\nXGBoost with Class Weights Results:")
    evaluate_model(xgb_model, X_valid_scaled, y_valid, model_type='xgboost')
    
    print("\n" + "="*50)
    print("APPROACH 7: XGBOOST WITH SMOTE")
    print("="*50)
    
    # XGBoost with SMOTE data
    xgb_smote = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_smote.fit(X_train_smote, y_train_smote)
    
    print("\nXGBoost with SMOTE Results:")
    evaluate_model(xgb_smote, X_valid_scaled, y_valid, model_type='xgboost')
    
    print("\n" + "="*50)
    print("APPROACH 8: XGBOOST HYPERPARAMETER TUNING")
    print("="*50)
    
    # XGBoost with different hyperparameters for better performance
    xgb_tuned = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_tuned.fit(X_train_scaled, y_train)
    
    print("\nXGBoost Tuned Results:")
    evaluate_model(xgb_tuned, X_valid_scaled, y_valid, model_type='xgboost')
    
    # Print feature importance for XGBoost
    print("\nTop 10 Most Important Features (XGBoost):")
    feature_importance = xgb_tuned.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    for i, feat_idx in enumerate(top_features):
        print(f"{i+1}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
    
    # Find optimal threshold for the best model
    best_models = {
        'Weighted LR': model_weighted,
        'SMOTE LR': model_smote,
        'ADASYN LR': model_adasyn,
        'SMOTEENN LR': model_smoteenn,
        'Random Forest': rf_model,
        'XGBoost Weighted': xgb_model,
        'XGBoost SMOTE': xgb_smote,
        'XGBoost Tuned': xgb_tuned
    }
    
    best_f1 = 0
    best_model_name = None
    best_model = None
    
    for name, model in best_models.items():
        model_type = 'xgboost' if 'XGBoost' in name else 'sklearn'
        
        if model_type == 'xgboost':
            y_probs = model.predict_proba(X_valid_scaled)[:, 1]
        else:
            y_probs = model.predict_proba(X_valid_scaled)[:, 1]
            
        optimal_threshold = find_optimal_threshold(y_valid, y_probs)
        
        print(f"\n{name} - Optimal threshold: {optimal_threshold:.3f}")
        _, _, f1 = evaluate_model(model, X_valid_scaled, y_valid, optimal_threshold, model_type)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
    
    print(f"\nBest model: {best_model_name} with F1-score: {best_f1:.3f}")
    
    # Generate final test predictions and choose best model
    model_type = 'xgboost' if 'XGBoost' in best_model_name else 'sklearn'
    
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
    
    output_df.to_csv("test_predictions_smiles_bert_balanced.csv", index=False)
    print(f"\nTest predictions saved to test_predictions_smiles_bert_balanced.csv")
    print(f"Used model: {best_model_name}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")

if __name__ == "__main__":
    main()