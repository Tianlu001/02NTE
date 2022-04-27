import pandas as pd
import numpy as np
from matrix_vec import *
from bags_vec import *



def read_data(xlsx_file, sheet_num): 
    data = pd.read_excel(xlsx_file, skipfooter=1, sheet_name=sheet_num)
    mol_mass = np.zeros(len(data))
    num_mols = len(data)
    for i, refcode in enumerate(data['KOJIN ID']):
        filename = '../molxyz/'+refcode+'.xyz'
        molecule = atom_type_list(filename)
        mass = molecule[0] * 12.011 + molecule[1] * 1.00797 + molecule[2]* 14.0067 + molecule[3] * 15.9994
        atoms = molecule.sum()
        mol_mass[i] = mass
    return data, mol_mass, num_mols


max_atoms = 70
##Determine target property




def generate_bags(data, type):
    X_summedBoH = []
    for i, refcode in enumerate(data['KOJIN ID']):
        filename = '../molxyz/'+refcode+'.xyz'
        summed_BoH_feature_names, summedBoH = type(filename)
 
        X_summedBoH += [summedBoH]
    return X_summedBoH


def generate_eigen(data, type):
    X_Vmat_eigs = np.zeros((num_mols, max_atoms))
    
    for i, refcode in enumerate(data['KOJIN ID']):
        filename = '../molxyz/'+refcode+'.xyz'
        this_atom_type_vec = atom_type_list(filename)
        this_volu_Cmat_eigs, this_volu_Cmat_as_vec = houlombmat3_and_eigenvalues_as_vec(filename, max_atoms )
    
    
        X_Vmat_eigs[i,:] = this_volu_Cmat_eigs
    return X_Vmat_eigs

#X_HBmat_eigs = np.concatenate((X_Hmat_eigs, X_Bmat_eigs),axis=1)

################## Train dataset ################################
data, mol_mass, num_mols = read_data('../list/mD5train2.xlsx',0)
X_summedBoH = generate_bags(data,summed_bag_of_heat3)

enthalpy_s = data['Solid enthalpy'].values
y_target = enthalpy_s

x_train = X_summedBoH
y_train = y_target

################## Test dataset ################################
data2, mol_mass2, num_mols2 = read_data('../list/mD5test2.xlsx',0)
X2_summedBoH = generate_bags(data2, summed_bag_of_heat3)

enthalpy_s2 = data2['Solid enthalpy'].values
y2_target = enthalpy_s2

x_validation = X2_summedBoH
y_validation = y2_target
