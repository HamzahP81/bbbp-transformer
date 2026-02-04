import deepchem as dc
import pandas as pd

def load_bbb_data():
    # Load your CSV file
    df = pd.read_csv('cleaned_BBBP.csv', sep='\t')
    
    # Map BBB labels to binary
    df['y'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
    df['ids'] = df['SMILES']
    df['w'] = 1.0
    
    # Create DeepChem dataset
    dataset = dc.data.DiskDataset.from_dataframe(
        df[['ids', 'y', 'w']], 
        data_dir='temp_bbb_data'
    )
    
    # Apply scaffold splitter
    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
    
    # Apply ECFP featurizer
    featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)
    train_dataset = featurizer.featurize(train_dataset)
    valid_dataset = featurizer.featurize(valid_dataset)
    test_dataset = featurizer.featurize(test_dataset)
    
    # Convert to dataframes
    train_df = train_dataset.to_dataframe()
    valid_df = valid_dataset.to_dataframe()
    test_df = test_dataset.to_dataframe()
    
    # Remove useless features (same as your original code)
    non_feature_cols = ['ids', 'w', 'y']
    feature_cols = [col for col in train_df.columns if col not in non_feature_cols]
    
    all_data = pd.concat([train_df[feature_cols], valid_df[feature_cols], test_df[feature_cols]])
    useful_features = all_data.columns[(all_data.sum(axis=0) > 0)]

    train_df = train_df[non_feature_cols + list(useful_features)]
    valid_df = valid_df[non_feature_cols + list(useful_features)]
    test_df = test_df[non_feature_cols + list(useful_features)]
    
    # Create raw dataframe
    raw_df = df[['ids', 'y', 'w']].copy()
    raw_df['X'] = raw_df['ids'].apply(lambda x: dc.utils.ConversionUtils.get_mol_from_smiles(x))

    print("Train set shape:", train_df.shape)
    print("Validation set shape:", valid_df.shape)
    print("Test set shape:", test_df.shape)
    print("Raw data shape:", raw_df.shape)
    print("\nSample SMILES from raw data:")
    print(raw_df[['ids', 'y']].head())
    
    # Save datasets as CSVs
    train_df.to_csv("train_bbbp.csv", index=False)
    valid_df.to_csv("valid_bbbp.csv", index=False)
    test_df.to_csv("test_bbbp.csv", index=False)
    raw_df.to_csv("raw_bbbp.csv", index=False)
    
    print("- train_bbbp.csv")
    print("- valid_bbbp.csv")
    print("- test_bbbp.csv")
    print("- raw_bbbp.csv")
    
    return train_df, valid_df, test_df, raw_df

if __name__ == "__main__":
    load_bbb_data()