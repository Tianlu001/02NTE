from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem

df = pd.read_csv('F1data.csv')
smiles = df['SMILES string']
name = df['Name']
print(name)


for i, smile in enumerate(smiles):
    nid = str(i+1).zfill(2)
    sid = name[i]
    cid = nid+sid
    mol= Chem.MolFromSmiles(smile)
    mol= Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(mol)
    
    writer = Chem.SDWriter(cid+'.sdf')
    writer.write(mol)
    writer.close()






