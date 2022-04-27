from rdkit import Chem
import sys

inputpdb = sys.argv[1]
outputsdf = sys.argv[2]

mol = Chem.MolFromPDBFile(inputpdb, removeHs=False)

writer = Chem.SDWriter(outputsdf)
writer.write(mol)
writer.close()

