import numpy as np
import copy
#from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem.rdmolops import Get3DDistanceMatrix, GetAdjacencyMatrix, GetDistanceMatrix
from rdkit.Chem.Graphs import CharacteristicPolynomial
from rdkit.Chem.Descriptors import _descList
from collections import defaultdict

atom_num_dict = {'C':6,'N':7,'O':8,'H':1,'F':9, 'Cl': 17, 'S': 16 }

atom_heat_dict = {'C': 11.400, 'H':-13.96, 'N':24.98, 'O':-14.500,'F':-30.0, 'Cl': 35.5, 'S': 32 }           #2nd version
atom_heat2_dict = {'C':15.200, 'H':4.700, 'N':20.000, 'O':16.900, 'F':15.26, 'Cl': 35.5, 'S': 32 }   #H20        
atom_heatr_dict = {'C':2.200, 'N':3.400, 'O':1.500, 'H':0.800, 'F':3.15, 'Cl': 35.453, 'S': 32.06 }

#----------------------------------------------------------------------------

def atom_type_list(filename):
    """
    returns a vector stating numbers of atoms [C, H, N, O] of a molecule
    Args:
        filename : (string) the .xyz input filename for the molecule
    Returns:
        atom_type_vec as Numpy arrays
    """
    xyzfile = open(filename, 'r')
    num_atoms_file = int(xyzfile.readline())
    xyzfile.close()
    atom_type_vec = np.zeros(5)
    atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
    atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]

    for i in range(num_atoms_file):
        if atom_symbols[i] == 'C':
            atom_type_vec[0] += 1     
        elif atom_symbols[i] == 'H':
            atom_type_vec[1] += 1     
        elif atom_symbols[i] == 'N':
            atom_type_vec[2] += 1     
        elif atom_symbols[i] == 'O':
            atom_type_vec[3] += 1     
        elif atom_symbols[i] == 'F':
            atom_type_vec[4] += 1     

    return atom_type_vec

#----------------------------------------------------------------------------
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
    atom_types = ['C', 'N', 'O', 'F', 'H']
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


#----------------------------------------------------------------------------
def sum_over_bonds(mol_list, predefined_bond_types=[], return_names=True):
    '''
        "Sum over bonds" aka "literal bag of bonds" aka "bond counting featurization"
        Note: Bond types are labeled according convention where the atom of the left is alphabetically less than
        the atom on the right. For instance, 'C=O' and 'O=C' bonds are lumped together under 'C=O', and NOT 'O=C'.
    Args:
        mol_list : a single mol object or list/iterable containing the RDKit mol objects for all of the molecules.
    Returns:
        bond_types : a list of strings describing the bond types in the feature vector
        X_LBoB : a NumPy array containing the feature vectors of shape (num_mols, num_bond_types)

    TODO: This code could be cleaned up substantially since we are using defaultdict now.
          <DCE 2018-06-12>
    '''

    if (isinstance(mol_list, list) == False):
        mol_list = [mol_list]

    empty_bond_dict = defaultdict(lambda : 0)
    num_mols = len(mol_list)

    if (len(predefined_bond_types) == 0 ):
        #first pass through to enumerate all bond types in all molecules and set them equal to zero in the dict
        for i, mol in enumerate(mol_list):
            bonds = mol.GetBonds()
            for bond in bonds:
                bond_start_atom = bond.GetBeginAtom().GetSymbol()
                bond_end_atom = bond.GetEndAtom().GetSymbol()
                bond_type = bond.GetSmarts(allBondsExplicit=True)
                bond_atoms = [bond_start_atom, bond_end_atom]
                if (bond_type == ''):
                    bond_type = "-"
                bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
                empty_bond_dict[bond_string] = 0
    else:
        for bond_string in predefined_bond_types:
            empty_bond_dict[bond_string] = 0

    #second pass through to construct X
    bond_types = list(empty_bond_dict.keys())
    num_bond_types = len(bond_types)

    X_LBoB = np.zeros([num_mols, num_bond_types])

    for i, mol in enumerate(mol_list):
        bonds = mol.GetBonds()
        bond_dict = copy.deepcopy(empty_bond_dict)
        for bond in bonds:
            bond_start_atom = bond.GetBeginAtom().GetSymbol()
            bond_end_atom = bond.GetEndAtom().GetSymbol()
            #skip dummy atoms
            if (bond_start_atom=='*' or bond_end_atom=='*'):
                pass
            else:
                bond_type = bond.GetSmarts(allBondsExplicit=True)
                if (bond_type == ''):
                    bond_type = "-"
                bond_atoms = [bond_start_atom, bond_end_atom]
                bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
                bond_dict[bond_string] += 1

        #at the end, pick out only the relevant ones
        X_LBoB[i,:] = [bond_dict[bond_type] for bond_type in bond_types]

    if (return_names):
        return bond_types, X_LBoB
    else:
        return X_LBoB

#----------------------------------------------------------------------------
