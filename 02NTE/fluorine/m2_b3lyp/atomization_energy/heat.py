import pandas as pd
import numpy as np
from matrix_vec import *
from bags_vec import *


def xyz_to_smiles(fname: str) -> str:
    
    mol = next(pybel.readfile("xyz", fname))

    smi = mol.write(format="smi")

    return smi.split()[0].strip()



#Read the data
data0 = pd.read_csv('../list/Htotalfluo_pwpb95.csv')
#data = data0.sample(n=300,replace=False)
data = data0

mol_mass = np.zeros(len(data))
num_atoms = np.zeros(len(data))
num_mols = len(data)

for i, refcode in enumerate(data['KOJIN ID']):
    filename = '../molxyz/'+refcode+'.xyz'
    molecule = atom_type_list(filename)
    mass = molecule[0] * 12.011 + molecule[1] * 1.00797 + molecule[2]* 14.0067 + molecule[3] * 15.9994
    atoms = molecule.sum()
    num_atoms[i] = atoms
    mol_mass[i] = mass

#print(mol_mass)
#print(num_atoms)
#print(num_mols)

#density = data['Density'].values
#mol_vol= mol_mass/density * 1.66

#print(mol_vol)


##Determine target property
#target_prop = 'Gas enthalpy'
target_prop = 'DFT enthalpy'

y1 = data[target_prop].values


#Number of atoms in largetst molecule : max_atoms
#max_atoms = int(max(num_atoms)) 
max_atoms = 70
#print(data['KOJIN ID'])
#print('Max atoms: %d' % max_atoms)

#Generate Coulomb Matrix
X_Atype_list = np.zeros((num_mols,5))
X_Cmat_eigs = np.zeros((num_mols, max_atoms))
X_Hmat_eigs = np.zeros((num_mols, max_atoms))
X_H2mat_eigs = np.zeros((num_mols, max_atoms))
X_summedBoB = []
X_summedBoH = []
X_summedBoH2 = []

filename_list = []

for i, refcode in enumerate(data['KOJIN ID']):
    filename = '../molxyz/'+refcode+'.xyz'
    this_atom_type_vec = atom_type_list(filename)
   #this_atom_Cmat_eigs, this_atom_Cmat_as_vec = coulombmat_and_eigenvalues_as_vec(filename, max_atoms )
   #this_heat_Cmat_eigs, this_heat_Cmat_as_vec = houlombmat_and_eigenvalues_as_vec(filename, max_atoms )
   #this_heat2_Cmat_eigs, this_heat2_Cmat_as_vec = houlombmat2_and_eigenvalues_as_vec(filename, max_atoms )
   #summed_BoB_feature_names, summedBoB = summed_bag_of_bonds(filename)
   #summed_BoH_feature_names, summedBoH = summed_bag_of_heats(filename)
   #summed_BoH2_feature_names, summedBoH2 = summed_bag_of_heat2(filename)
   #summed_BoH2_feature_names, summedBoH2 = summed_bag_of_heat3(filename)

    filename_list += [filename]

    X_Atype_list[i,:] = this_atom_type_vec
   #X_Cmat_eigs[i,:] = this_atom_Cmat_eigs
   #X_Hmat_eigs[i,:] = this_heat_Cmat_eigs
   #X_H2mat_eigs[i,:] = this_heat2_Cmat_eigs
   #X_summedBoB += [summedBoB]
   #X_summedBoH += [summedBoH]
   #X_summedBoH2 += [summedBoH2]


#X_HBmat_eigs = np.concatenate((X_Hmat_eigs, X_Bmat_eigs),axis=1)


#x_train = X_H2mat_eigs 
#x_train = X_summedBoB 
#x_train = X_summedBoH 
x_train = X_Atype_list

y_train = y1 

#print(x_train)
