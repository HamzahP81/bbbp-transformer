import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split

def load_pubchem_bbb_data():
    
    # Load the TSV file (adjust path as needed)
    df = pd.read_csv('B3DB_classification.tsv', sep='\t')
    
    # Map BBB labels to binary (1 for BBB+, 0 for BBB-)
    df['y'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
    
    # Use SMILES as ids
    df['ids'] = df['SMILES']
    
    # Calculate weights (using uniform weights like your BBBP, or could use class balancing)
    df['w'] = 1.0  # You can adjust this based on your needs
    
    # Split based on existing groups (A, B, C)
    train_df = df[df['group'] == 'B'].copy()
    valid_df = df[df['group'] == 'C'].copy()
    test_df = df[df['group'] == 'A'].copy()
    
    # Generate ECFP features to match BBBP format
    def generate_ecfp_features(smiles_list, radius=2, nBits=1024):
        features = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits)
                features.append(list(fp))
            else:
                features.append([0] * nBits)
        return np.array(features)
    
    # Generate ECFP features for each split
    train_features = generate_ecfp_features(train_df['ids'].tolist())
    valid_features = generate_ecfp_features(valid_df['ids'].tolist())
    test_features = generate_ecfp_features(test_df['ids'].tolist())
    
    # Add feature columns to dataframes
    feature_cols = [f'ecfp_{i}' for i in range(train_features.shape[1])]
    
    train_features_df = pd.DataFrame(train_features, columns=feature_cols, index=train_df.index)
    valid_features_df = pd.DataFrame(valid_features, columns=feature_cols, index=valid_df.index)
    test_features_df = pd.DataFrame(test_features, columns=feature_cols, index=test_df.index)
    
    # Combine with metadata
    train_df = pd.concat([train_df[['ids', 'y', 'w']], train_features_df], axis=1)
    valid_df = pd.concat([valid_df[['ids', 'y', 'w']], valid_features_df], axis=1)
    test_df = pd.concat([test_df[['ids', 'y', 'w']], test_features_df], axis=1)
    
    # Remove useless features (columns with all zeros)
    print("Removing useless features...")
    all_data = pd.concat([train_features_df, valid_features_df, test_features_df])
    useful_features = all_data.columns[(all_data.sum(axis=0) > 0)]
    
    # Keep only useful features
    train_df = train_df[['ids', 'y', 'w'] + list(useful_features)]
    valid_df = valid_df[['ids', 'y', 'w'] + list(useful_features)]
    test_df = test_df[['ids', 'y', 'w'] + list(useful_features)]
    
    # Create raw dataframe (matches your BBBP raw format)
    raw_df = df[['ids', 'y', 'w']].copy()
    
    # Add RDKit Mol objects for compatibility
    raw_df['X'] = raw_df['ids'].apply(lambda x: Chem.MolFromSmiles(x))
    
    # Print statistics
    print("PubChem BBB dataset loaded successfully!")
    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {valid_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Raw data shape: {raw_df.shape}")
    print(f"\nClass distribution:")
    print(f"Train - BBB+: {train_df['y'].sum()}, BBB-: {len(train_df) - train_df['y'].sum()}")
    print(f"Valid - BBB+: {valid_df['y'].sum()}, BBB-: {len(valid_df) - valid_df['y'].sum()}")
    print(f"Test - BBB+: {test_df['y'].sum()}, BBB-: {len(test_df) - test_df['y'].sum()}")
    
    print("\nSample SMILES from raw data:")
    print(raw_df[['ids', 'y']].head())
    
    # Save datasets as CSVs (same naming as BBBP)
    train_df.to_csv("train_bbbp.csv", index=False)
    valid_df.to_csv("valid_bbbp.csv", index=False)
    test_df.to_csv("test_bbbp.csv", index=False)
    raw_df.to_csv("raw_bbbp.csv", index=False)
    
    print("\nDatasets saved as CSV files:")
    print("- train_bbbp.csv")
    print("- valid_bbbp.csv")
    print("- test_bbbp.csv")
    print("- raw_bbbp.csv")
    
    return train_df, valid_df, test_df, raw_df

if __name__ == "__main__":
    load_pubchem_bbb_data()