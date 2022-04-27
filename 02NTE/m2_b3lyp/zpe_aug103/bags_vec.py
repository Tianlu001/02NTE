import numpy as np
import copy
#from sklearn.preprocessing import StandardScaler
#from rdkit import Chem
#from rdkit.Chem.rdmolops import Get3DDistanceMatrix, GetAdjacencyMatrix, GetDistanceMatrix
#from rdkit.Chem.Graphs import CharacteristicPolynomial
#from rdkit.Chem.Descriptors import _descList
#from collections import defaultdict
atom_num_dict = {'C':6,'N':7,'O':8,'H':1,'F':9, 'Cl': 17, 'S': 16 }

atom_heat_dict = {'C': 11.400, 'H':-13.96, 'N':24.98, 'O':-14.500,'F':0.0, 'Cl': 35.5, 'S': 32 }           #2nd version
atom_heat2_dict = {'C':15.200, 'H':4.700, 'N':20.000, 'O':16.900, 'F':1.0, 'Cl': 35.5, 'S': 32 }   #H20        
atom_heatr_dict = {'C':2.200, 'N':3.400, 'O':1.500, 'H':0.800, 'F':18.998403, 'Cl': 35.453, 'S': 32.06 }

#----------------------------------------------------------------------------
def summed_bag_of_bonds(filename):
    """
        Based on   Hansen, et al., The Journal of Physical Chemistry Letters 2015 6 (12), 2326-2331
        DOI: 10.1021/acs.jpclett.5b00831, URL: http://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b00831
        However, the Coulomb matrix terms for each atom pair (C-C, C-N, C-O, etc) are **summed** together.
        The diagonal terms of the Coulomb matrix are concatenated with the resulting vector.
        So the resulting feature vector for each molecule is a vector of length
        (num_atom_pair_types + num_atom_types). This is different than the original BoB, which maintains each
        CM entry in the feature vector.
    Args:
        filename : (string) the .xyz input filename for the molecule
    Returns:
        (feature_names, BoB_list) as lists
    """
    xyzfile = open(filename, 'r')
    num_atoms_file = int(xyzfile.readline())
    xyzfile.close()
    Cmat = np.zeros((num_atoms_file,num_atoms_file))
    chargearray = np.zeros((num_atoms_file, 1))
    xyzmatrix = np.loadtxt(filename, skiprows=2, usecols=[1,2,3])
    atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
    atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]
    chargearray = [atom_num_dict[symbol] for symbol in atom_symbols]

    #------- initialize dictionary for storing each bag ---------
    atom_types = ['C', 'N', 'O', 'F', 'H']
    num_atom_types = len(atom_types)

    BoB_dict = {}
    for atom_type in atom_types:
        BoB_dict[atom_type] = 0

    for i in range(num_atom_types):
        for j in range(i,num_atom_types):
            BoB_dict[atom_types[i]+atom_types[j]] = 0

    #------- populate BoB dict -----------------------------------
    for i in range(num_atoms_file):
        for j in range(i, num_atoms_file):
            if i == j:
                BoB_dict[atom_symbols[i]] += 0.5*chargearray[i]**2.4
            else:
                dict_key = atom_symbols[i]+atom_symbols[j]
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                CM_term = chargearray[i]*chargearray[j]/dist
                try:
                    BoB_dict[dict_key] += CM_term
                except KeyError:
                    dict_key = atom_symbols[j]+atom_symbols[i]
                    BoB_dict[dict_key] += CM_term

    #------- process into list -------------------------------------
    feature_names = list(BoB_dict.keys())
    BoB_list = [BoB_dict[feature] for feature in feature_names]

    return feature_names, BoB_list

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def summed_bag_of_heat1(filename):
    """
        Based on   Hansen, et al., The Journal of Physical Chemistry Letters 2015 6 (12), 2326-2331
        DOI: 10.1021/acs.jpclett.5b00831, URL: http://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b00831
        However, the Coulomb matrix terms for each atom pair (C-C, C-N, C-O, etc) are **summed** together.
        The diagonal terms of the Coulomb matrix are concatenated with the resulting vector.
        So the resulting feature vector for each molecule is a vector of length
        (num_atom_pair_types + num_atom_types). This is different than the original BoB, which maintains each
        CM entry in the feature vector.
    Args:
        filename : (string) the .xyz input filename for the molecule
    Returns:
        (feature_names, BoB_list) as lists
    """
    xyzfile = open(filename, 'r')
    num_atoms_file = int(xyzfile.readline())
    xyzfile.close()
    Cmat = np.zeros((num_atoms_file,num_atoms_file))
    chargearray = np.zeros((num_atoms_file, 1))
    xyzmatrix = np.loadtxt(filename, skiprows=2, usecols=[1,2,3])
    atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
    atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]
    chargearray = [atom_heat_dict[symbol] for symbol in atom_symbols]

    #------- initialize dictionary for storing each bag ---------
    atom_types = ['C', 'N', 'O', 'H']
    num_atom_types = len(atom_types)
    H_corr = 20

    BoB_dict = {}
    for atom_type in atom_types:
        BoB_dict[atom_type] = 0

    for i in range(num_atom_types):
        for j in range(i,num_atom_types):
            BoB_dict[atom_types[i]+atom_types[j]] = 0

    #------- populate BoB dict -----------------------------------
    for i in range(num_atoms_file):
        for j in range(i, num_atoms_file):
            if i == j:
                BoB_dict[atom_symbols[i]] += 1.0*chargearray[i]**1.0
            else:
                dict_key = atom_symbols[i]+atom_symbols[j]
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                CM_term = 0.0*(chargearray[i]*chargearray[j])**1/dist**2.0
                try:
                    BoB_dict[dict_key] += CM_term
                except KeyError:
                    dict_key = atom_symbols[j]+atom_symbols[i]
                    BoB_dict[dict_key] += CM_term

    #------- process into list -------------------------------------
    feature_names = list(BoB_dict.keys())
    BoB_list = [BoB_dict[feature] for feature in feature_names]

    return feature_names, BoB_list

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def summed_bag_of_heat2(filename):
    """
        Based on   Hansen, et al., The Journal of Physical Chemistry Letters 2015 6 (12), 2326-2331
        DOI: 10.1021/acs.jpclett.5b00831, URL: http://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b00831
        However, the Coulomb matrix terms for each atom pair (C-C, C-N, C-O, etc) are **summed** together.
        The diagonal terms of the Coulomb matrix are concatenated with the resulting vector.
        So the resulting feature vector for each molecule is a vector of length
        (num_atom_pair_types + num_atom_types). This is different than the original BoB, which maintains each
        CM entry in the feature vector.
    Args:
        filename : (string) the .xyz input filename for the molecule
    Returns:
        (feature_names, BoB_list) as lists
    """
    xyzfile = open(filename, 'r')
    num_atoms_file = int(xyzfile.readline())
    xyzfile.close()
    Cmat = np.zeros((num_atoms_file,num_atoms_file))
    chargearray = np.zeros((num_atoms_file, 1))
    xyzmatrix = np.loadtxt(filename, skiprows=2, usecols=[1,2,3])
    atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
    atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]
    chargearray = [atom_heat_dict[symbol] for symbol in atom_symbols]
    dibondarray = [atom_heat2_dict[symbol] for symbol in atom_symbols]
    lengtharray = [atom_heatr_dict[symbol] for symbol in atom_symbols]

    #------- initialize dictionary for storing each bag ---------
    atom_types = ['C', 'N', 'O', 'H']
    num_atom_types = len(atom_types)

    BoB_dict = {}
    for atom_type in atom_types:
        BoB_dict[atom_type] = 0

    for i in range(num_atom_types):
        for j in range(i,num_atom_types):
            BoB_dict[atom_types[i]+atom_types[j]] = 0

    #------- populate BoB dict -----------------------------------
    for i in range(num_atoms_file):
        for j in range(i, num_atoms_file):
            if i == j:
                BoB_dict[atom_symbols[i]] += 1.0*chargearray[i]**1.0
               #BoB_dict[atom_symbols[i]] += 1.0**2.0
            else:
                dict_key = atom_symbols[i]+atom_symbols[j]
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                CM_term = -0.2*(dibondarray[i])*(dibondarray[j])*np.power(dist,-(lengtharray[i]+lengtharray[j])/2)
                try:
                    BoB_dict[dict_key] += CM_term
                except KeyError:
                    dict_key = atom_symbols[j]+atom_symbols[i]
                    BoB_dict[dict_key] += CM_term

    #------- process into list -------------------------------------
    feature_names = list(BoB_dict.keys())
    BoB_list = [BoB_dict[feature] for feature in feature_names]

    return feature_names, BoB_list

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def summed_bag_of_heat3(filename):
    """
        Based on   Hansen, et al., The Journal of Physical Chemistry Letters 2015 6 (12), 2326-2331
        DOI: 10.1021/acs.jpclett.5b00831, URL: http://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b00831
        However, the Coulomb matrix terms for each atom pair (C-C, C-N, C-O, etc) are **summed** together.
        The diagonal terms of the Coulomb matrix are concatenated with the resulting vector.
        So the resulting feature vector for each molecule is a vector of length
        (num_atom_pair_types + num_atom_types). This is different than the original BoB, which maintains each
        CM entry in the feature vector.
    Args:
        filename : (string) the .xyz input filename for the molecule
    Returns:
        (feature_names, BoB_list) as lists
    """
    xyzfile = open(filename, 'r')
    num_atoms_file = int(xyzfile.readline())
    xyzfile.close()
    Cmat = np.zeros((num_atoms_file,num_atoms_file))
    chargearray = np.zeros((num_atoms_file, 1))
    xyzmatrix = np.loadtxt(filename, skiprows=2, usecols=[1,2,3])
    atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
    atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]
    chargearray = [atom_heat_dict[symbol] for symbol in atom_symbols]
    dibondarray = [atom_heat2_dict[symbol] for symbol in atom_symbols]
    lengtharray = [atom_heatr_dict[symbol] for symbol in atom_symbols]

    #------- initialize dictionary for storing each bag ---------
    atom_types = ['C', 'N', 'O', 'H']
    num_atom_types = len(atom_types)

    BoB_dict = {}
    for atom_type in atom_types:
        BoB_dict[atom_type] = 0

    for i in range(num_atom_types):
        for j in range(i,num_atom_types):
            BoB_dict[atom_types[i]+atom_types[j]] = 0

    #------- populate BoB dict -----------------------------------
    for i in range(num_atoms_file):
        for j in range(i, num_atoms_file):
            if i == j:
                BoB_dict[atom_symbols[i]] += 1.0*chargearray[i]**1.0
               #BoB_dict[atom_symbols[i]] += 1.0**2.0
            else:
                dict_key = atom_symbols[i]+atom_symbols[j]
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                CM_term = -0.2*(dibondarray[i])*(dibondarray[j])*np.power(dist,-(lengtharray[i]+lengtharray[j])/2)/(1+np.exp(2*(dist-1.8))) 
                try:
                    BoB_dict[dict_key] += CM_term
                except KeyError:
                    dict_key = atom_symbols[j]+atom_symbols[i]
                    BoB_dict[dict_key] += CM_term

    #------- process into list -------------------------------------
    feature_names = list(BoB_dict.keys())
    BoB_list = [BoB_dict[feature] for feature in feature_names]

    return feature_names, BoB_list

#----------------------------------------------------------------------------
