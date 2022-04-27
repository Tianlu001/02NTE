import pandas as pd
import numpy as np
from matrix_vec import *
from bags_vec import *



def read_data(xlsx_file, sheet_num): 
    data = pd.read_excel(xlsx_file, sheet_name=sheet_num)
    mol_mass = np.zeros(len(data))
    num_mols = len(data)
    for i, refcode in enumerate(data['KOJIN ID']):
        filename = '../molxyz/'+refcode+'.xyz'
        molecule = atom_type_list(filename)
        mass = molecule[0] * 12.011 + molecule[1] * 1.00797 + molecule[2]* 14.0067 + molecule[3] * 15.9994
        atoms = molecule.sum()
        mol_mass[i] = mass
    return data, mol_mass, num_mols


# Estimated data
data, mol_mass, num_mols = read_data('../list/Hnzpe_b3lypD3.xlsx', 0)
print(data['KOJIN ID'])
#target_cep= np.array(data['Nzpe energy'].values)
#X_summedBoH = generate_bags(data,summed_bag_of_heat3)
#X_PlusCeP = np.insert(X_summedBoH, 0, values=target_cep, axis=1)

