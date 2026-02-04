import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)

def get_ecfp_features(mol, n_bits=1024):
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits), dtype=int)

# Load cleaned BBBP data
df = pd.read_csv("cleaned_bbbp.csv")

# Map BBB labels to binary
df['y'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
df['ids'] = df['SMILES']
df['w'] = 1.0

# Convert SMILES to Mol and compute fingerprints
df["mol"] = df["SMILES"].apply(mol_from_smiles)
df["features"] = df["mol"].apply(get_ecfp_features)

# Stack fingerprint features into 1024 columns
features_array = np.stack(df["features"].values)
features_df = pd.DataFrame(features_array, columns=[f"f_{i}" for i in range(1024)])

# Combine with labels and SMILES
df_final = pd.concat([features_df, df[["y", "w", "ids"]]], axis=1)

# Train/val/test split (80/10/10)
train_val_df, test_df = train_test_split(df_final, test_size=0.2, stratify=df_final["y"], random_state=42)
train_df, valid_df = train_test_split(train_val_df, test_size=0.1111, stratify=train_val_df["y"], random_state=42)

# Save splits
train_df.to_csv("train_bbbp.csv", index=False)
valid_df.to_csv("valid_bbbp.csv", index=False)
test_df.to_csv("test_bbbp.csv", index=False)

print("Train set shape:", train_df.shape)
print("Validation set shape:", valid_df.shape)
print("Test set shape:", test_df.shape)
    
# Save datasets as CSVs
train_df.to_csv("train_bbbp.csv", index=False)
valid_df.to_csv("valid_bbbp.csv", index=False)
test_df.to_csv("test_bbbp.csv", index=False)

print("- train_bbbp.csv")
print("- valid_bbbp.csv")
print("- test_bbbp.csv")
    
