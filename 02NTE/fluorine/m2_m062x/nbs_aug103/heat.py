import pandas as pd
import numpy as np
from bags_vec import *
from rdkit import Chem
from rdkit.Chem import Descriptors



def read_data(xlsx_file): 
    data = pd.read_csv(xlsx_file)
    mol_mass = np.zeros(len(data))
    num_mols = len(data)
    kojin_id = []
    for i, refcode in enumerate(data['KOJIN ID']):
        filename = '../molxyz/'+refcode+'.xyz'
        molecule = atom_type_list(filename)
        mass = molecule[0] * 12.011 + molecule[1] * 1.00797 + molecule[2]* 14.0067 + molecule[3] * 15.9994
        atoms = molecule.sum()
        mol_mass[i] = mass
        kojin_id.append(refcode)
    return data, mol_mass, num_mols, kojin_id

def read_id(kojin_id): 
    molecule_list=[]
    for i, refcode in enumerate(kojin_id):
        filename = '../molsdf/'+refcode+'.sdf'
        suppl = Chem.SDMolSupplier(filename)
        mols = [Chem.MolToSmiles(mol) for mol in suppl if mol]
        molecule_list.extend(mols)
   #molecule_list= [Chem.AddHs(Chem.MolFromSmiles(mol)) for mol in molecule_list]
    molecule_list= [Chem.MolFromSmiles(mol) for mol in molecule_list]
    return molecule_list

max_atoms = 50

##Determine target property

def generate_bags(data, type):
    X_summedBoH = []
    for i, refcode in enumerate(data['KOJIN ID']):
        filename = '../molxyz/'+refcode+'.xyz'
        summed_BoH_feature_names, summedBoH = type(filename)
 
        X_summedBoH += [summedBoH]
    return X_summedBoH

def generate_sob(mol_list):
    bond_types=['N-O','N:O','N-N','N=O','N=N','N:N','N#N','C-N','C-C','C-H','C:N','C:C','C-O','C=O','C=N','C=C','H-O','H-N','C-F','F-N']
    bondtype,X_SOB=sum_over_bonds(mol_list,predefined_bond_types=bond_types)
    return X_SOB



# Estimated data
data, mol_mass, num_mols, kojin_id = read_data('../list/Hnzpefluo_m062xD3.csv')
mol_list= read_id(kojin_id)

target_cep= np.array(data['Nzpe energy'].values)

X_summedBoH = generate_bags(data,summed_bag_of_heat3)
X_SOB = generate_sob(mol_list)
X_BS = np.concatenate((X_SOB, X_summedBoH),axis=1)
X_PlusCeP = np.insert(X_BS, 0, values=target_cep, axis=1)

# benchmark data
data2, mol_mass2, num_mols2, kojin_id = read_data('../list/Hnzpefluo_pwpb95.csv')
enthalpy = data2['DFT enthalpy'].values
y = enthalpy


x_train = X_PlusCeP
y_train = y

#print("['N-O','N:O','N-N','N=O','N=N','N:N','N#N','C-N','C-C','C-H','C:N','C:C','C-O','C=O','C=N','C=C','H-O','H-N']")
#print(X_SOB[77])
