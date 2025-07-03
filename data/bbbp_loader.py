import deepchem as dc
import pandas as pd


tasks, datasets, transformers = dc.molnet.load_bbbp(
    featurizer='ECFP',
    splitter='scaffold',
    transformers=['balancing'],
    reload=False
)
train_dataset, valid_dataset, test_dataset = datasets

train_df = train_dataset.to_dataframe()
valid_df = valid_dataset.to_dataframe()
test_df = test_dataset.to_dataframe()
raw_bbbp = dc.molnet.load_bbbp(featurizer='Raw', splitter=None, reload=False)[1][0]
raw_df = raw_bbbp.to_dataframe()

# prview of datasets- comment out if not needed
print("BBBP dataset: ")
print("Train set shape:", train_df.shape)
print("Validation set shape:", valid_df.shape)
print("Test set shape:", test_df.shape)
print("\nSample SMILES from raw data:")
print(raw_df.head())


# save as CSVs- don't COMMENT OUT or CHANGE NAMES
train_df.to_csv("train_bbbp.csv", index=False)
valid_df.to_csv("valid_bbbp.csv", index=False)
test_df.to_csv("test_bbbp.csv", index=False)
raw_df.to_csv("raw_bbbp.csv", index=False)
