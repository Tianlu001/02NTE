import pandas as pd
import numpy as np
from matrix_vec import *
from bags_vec import *

import pickle
import joblib


def read_data(xlsx_file): 
    data = pd.read_excel(xlsx_file, skipfooter=1,sheet_nane=0)
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
    molecule_list= [Chem.AddHs(Chem.MolFromSmiles(mol)) for mol in molecule_list]
    return molecule_list

def generate_bags(data, type):
   #type=summed_bag_of_heat3
    X_summedBoB = []
    for i, refcode in enumerate(data['KOJIN ID']):
        filename = '../molxyz/'+refcode+'.xyz'
        summed_BoB_feature_names, summedBoB = type(filename)
 
        X_summedBoB += [summedBoB]
    return X_summedBoB


#############################
#Read the data

data, mol_mass, num_mols, kojin_id = read_data('../list/mD4Heaterror.xlsx')
#mol_list= read_id(kojin_id)

X_summedBoB = generate_bags(data,summed_bag_of_heat3)
X_summedBoB = np.array(X_summedBoB)
target= data['Solid enthalpy'].values

#target_cep= np.array(data['Size'].values)

#X_combine = np.concatenate((X_SOB, X_summedBoB),axis=1)
#X_PlusCeP = np.insert(X_combine, 0, values=target_cep, axis=1)


x_predict = X_summedBoB
y_target = target 

regr = joblib.load('KRR.joblib')
y_predict = regr.predict(x_predict)

y_compare = np.array((y_target, y_predict))
y_compare = np.transpose(y_compare)

print("validate error:", np.mean(((y_target- y_predict)**2)**0.5))


np.savetxt('xxx2.dat', y_compare, delimiter=',  ',fmt='%10.5f')
