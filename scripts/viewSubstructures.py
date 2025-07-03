from rdkit import Chem
# Check if RDKit is installed and working- can comment out or removed (for debugging)
mol = Chem.MolFromSmiles("CCO")
print(mol is not None) 

from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolsToGridImage
from collections import Counter, defaultdict
import pandas as pd

raw_df = pd.read_csv("raw_bbbp.csv")
radius = 2
nBits = 1024
feature_counts = Counter()
feature_mol_map = defaultdict(list)

for smi in raw_df['ids']:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    info = {}
    # Get Morgan fingerprint with atom information
    fp = AllChem.GetMorganFingerprint(mol, radius, useCounts=True, useFeatures=False, bitInfo=info)
    
    for feature_id, count in fp.GetNonzeroElements().items():
        feature_counts[feature_id] += count
        if feature_id in info:
            feature_mol_map[feature_id].append((mol, info[feature_id]))

# only get top 50 most frequent features
top_features = feature_counts.most_common(50)

decoded_features = []
for feature_id, freq in top_features:
    try:
        example_mol, atom_info = feature_mol_map[feature_id][0]
        center_atom_idx, rad = atom_info[0]
        env = Chem.FindAtomEnvironmentOfRadiusN(example_mol, rad, center_atom_idx)
        amap = {}
        submol = Chem.PathToSubmol(example_mol, env, atomMap=amap)
        smiles = Chem.MolToSmiles(submol)
        decoded_features.append({
            'feature_id': feature_id,
            'smiles': smiles,
            'frequency': freq
        })
    except Exception as e:
        print(f"Skipping feature {feature_id}: {e}")
        continue

# Convert to DataFrame and save
df_feats = pd.DataFrame(decoded_features)
df_feats['smiles'] = df_feats['smiles'].replace('', 'Unmapped')
df_feats.to_csv("top_decoded_features.csv", index=False)
print(df_feats.head())

# for visualisations of substructyres
mols = [Chem.MolFromSmiles(d['smiles']) for d in decoded_features if d['smiles']]
legends = [f"ID: {d['feature_id']}" for d in decoded_features]
img = MolsToGridImage(mols, molsPerRow=5, legends=legends)
img.save("top_features.png")
