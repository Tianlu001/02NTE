import pandas as pd
import numpy as np
from matrix_vec import *
from bags_vec import *

import pickle
import joblib


def read_data(xlsx_file): 
    data = pd.read_csv(xlsx_file)
    mol_mass = np.zeros(len(data))
    num_mols = len(data)
    for i, refcode in enumerate(data['KOJIN ID']):
        filename = '../sep531xyz/'+str(refcode).zfill(4)+'.xyz'
        molecule = atom_type_list(filename)
        mass = molecule[0] * 12.011 + molecule[1] * 1.00797 + molecule[2]* 14.0067 + molecule[3] * 15.9994
        atoms = molecule.sum()
        mol_mass[i] = mass
    return data, mol_mass, num_mols


def generate_bags(data, type):
    X_summedBoH = []
    for i, refcode in enumerate(data['KOJIN ID']):
        filename = '../sep531xyz/'+str(refcode).zfill(4)+'.xyz'
        summed_BoH_feature_names, summedBoH = type(filename)
 
        X_summedBoH += [summedBoH]
    return X_summedBoH


#############################
#Read the data

data, mol_mass, num_mols = read_data('../list/H531_m062x.csv')
target_cep= np.array(data['Nzpe energy'].values)
X_summedBoH = generate_bags(data,summed_bag_of_heat3)
X_PlusCeP = np.insert(X_summedBoH, 0, values=target_cep, axis=1)

# benchmark data
data2, mol_mass2, num_mols2 = read_data('../list/H531_pwpb95.csv')
enthalpy = data2['DFT enthalpy'].values
target = enthalpy


x_predict = X_PlusCeP
y_target = target 

regr = joblib.load('KRR.joblib')
y_predict = regr.predict(x_predict)

y_compare = np.array((y_target, y_predict, y_predict-y_target))
y_compare = np.transpose(y_compare)

print("validate error:", np.mean(((y_target- y_predict)**2)**0.5))


np.savetxt('xxx2.dat', y_compare, delimiter=',  ',fmt='%10.5f')
