import numpy as np
import copy
#from sklearn.preprocessing import StandardScaler
#from rdkit import Chem
#from rdkit.Chem.rdmolops import Get3DDistanceMatrix, GetAdjacencyMatrix, GetDistanceMatrix
#from rdkit.Chem.Graphs import CharacteristicPolynomial
#from rdkit.Chem.Descriptors import _descList
#from collections import defaultdict

""" Dictionaries"""
atom_num_dict = {'C':6,'N':7,'O':8,'H':1,'F':9, 'Cl': 17, 'S': 16 }
atom_mass_dict = {'C':12.011,'N':14.0067,'O':15.9994,'H':1.00797,'F':18.998403, 'Cl': 35.453, 'S': 32.06 }
atom_mass2_dict = {'C':26.30916, 'N':40.83586, 'O':41.12777, 'H':19.89951, 'F':18.998403, 'Cl': 35.453, 'S': 32.06 }  #M3 AR
atom_volu_dict = {'C':13.24, 'N':11.58, 'O':11.99, 'H':4.75, 'F':18.998403, 'Cl': 35.453, 'S': 32.06 }
#[13.24472125  4.750446   11.68239518 11.99330943]
atom_heat_dict = {'C':7.05958178, 'H':-10.1924, 'N':23.0005, 'O':-13.22317,'F':0.0, 'Cl': 35.5, 'S': 32 }   #H0        
#atom_heat_dict = {'C':-8.000, 'H':-7.000, 'N':24.000, 'O':-15.000,'F':0.0, 'Cl': 35.5, 'S': 32 }           #HL
atom_heat2_dict = {'C':5.000, 'H':-0.200, 'N':-1.400, 'O':-6.777, 'F':1.0, 'Cl': 35.5, 'S': 32 }   #H20        
atom_heatr_dict = {'C':-2.100, 'N':-1.800, 'O':-1.900, 'H':-1.400, 'F':18.998403, 'Cl': 35.453, 'S': 32.06 }

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
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
def moulombmat_and_eigenvalues_as_vec(filename, padded_size, sort=True):
    """
    returns Coulomb matrix and **sorted** Coulomb matrix eigenvalues
    Args:
        filename : (string) the .xyz input filename for the molecule
        padded_size : the number of atoms in the biggest molecule to be considered (same as padded eigenvalue vector length)
    Returns:
        (Eigenvalues vector, Coulomb matrix vector) as Numpy arrays
    """
    xyzfile = open(filename, 'r')
    num_atoms_file = int(xyzfile.readline())
    xyzfile.close()
    Cmat = np.zeros((num_atoms_file,num_atoms_file))
    chargearray = np.zeros((num_atoms_file, 1))
    xyzmatrix = np.loadtxt(filename, skiprows=2, usecols=[1,2,3])
    atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
    atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]
    chargearray = [atom_mass_dict[symbol] for symbol in atom_symbols]

    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if i == j:
                Cmat[i,j]=0.5*chargearray[i]**1.0   # Diagonal terms
            else:
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                Cmat[i,j]=0.1*chargearray[i]*chargearray[j]/dist   #Pair-wise repulsion

    Cmat_eigenvalues = np.linalg.eigvals(Cmat)

    if (sort): Cmat_eigenvalues = sorted(Cmat_eigenvalues, reverse=True) #sort

    Cmat_as_vec = []
    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if (j>=i):
                Cmat_as_vec += [Cmat[i,j]]

    pad_width = (padded_size**2 - padded_size)//2 + padded_size - ((num_atoms_file**2 - num_atoms_file)//2 + num_atoms_file)
    Cmat_as_vec = Cmat_as_vec + [0]*pad_width

    Cmat_as_vec = np.array(Cmat_as_vec)

    pad_width = padded_size - num_atoms_file
    Cmat_eigenvalues = np.pad(Cmat_eigenvalues, ((0, pad_width)), mode='constant')

    return Cmat_eigenvalues, Cmat_as_vec


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def houlombmat2_and_eigenvalues_as_vec(filename, padded_size, sort=True):
    """
    returns Coulomb matrix and **sorted** Coulomb matrix eigenvalues
    Args:
        filename : (string) the .xyz input filename for the molecule
        padded_size : the number of atoms in the biggest molecule to be considered (same as padded eigenvalue vector length)
    Returns:
        (Eigenvalues vector, Coulomb matrix vector) as Numpy arrays
    """
    xyzfile = open(filename, 'r')
    num_atoms_file = int(xyzfile.readline())
    xyzfile.close()
    Cmat = np.zeros((num_atoms_file,num_atoms_file))
    chargearray = np.zeros((num_atoms_file, 1))
    xyzmatrix = np.loadtxt(filename, skiprows=2, usecols=[1,2,3])
    atom_symbols = np.loadtxt(filename, skiprows=2, dtype=bytes, usecols=[0])
    atom_symbols = [symbol.decode('utf-8') for symbol in atom_symbols]
    chargearray = [atom_heat2_dict[symbol] for symbol in atom_symbols]
    dibondarray = [atom_heat2_dict[symbol] for symbol in atom_symbols]
    lengtharray = [atom_heatr_dict[symbol] for symbol in atom_symbols]

    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if i == j:
                Cmat[i,j]=1.0*chargearray[i]**1.0   # Diagonal terms
            else:
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                Cmat[i,j] = 1.0*(dibondarray[i]*dibondarray[j])**1.0/dist**2.0

    Cmat_eigenvalues = np.linalg.eigvals(Cmat)

    if (sort): Cmat_eigenvalues = sorted(Cmat_eigenvalues, reverse=True) #sort

    Cmat_as_vec = []
    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if (j>=i):
                Cmat_as_vec += [Cmat[i,j]]

    pad_width = (padded_size**2 - padded_size)//2 + padded_size - ((num_atoms_file**2 - num_atoms_file)//2 + num_atoms_file)
    Cmat_as_vec = Cmat_as_vec + [0]*pad_width

    Cmat_as_vec = np.array(Cmat_as_vec)

    pad_width = padded_size - num_atoms_file
    Cmat_eigenvalues = np.pad(Cmat_eigenvalues, ((0, pad_width)), mode='constant')

    return Cmat_eigenvalues, Cmat_as_vec


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def houlombmat_and_eigenvalues_as_vec(filename, padded_size, sort=True):
    """
    returns Coulomb matrix and **sorted** Coulomb matrix eigenvalues
    Args:
        filename : (string) the .xyz input filename for the molecule
        padded_size : the number of atoms in the biggest molecule to be considered (same as padded eigenvalue vector length)
    Returns:
        (Eigenvalues vector, Coulomb matrix vector) as Numpy arrays
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

    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if i == j:
                Cmat[i,j]=2.5*(chargearray[i]+0)**1.0   # Diagonal terms
            else:
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                Cmat[i,j]=0.6*((chargearray[i]+0)*(chargearray[j]+0))**2/dist**2.0   #Pair-wise repulsion

    Cmat_eigenvalues = np.linalg.eigvals(Cmat)

    if (sort): Cmat_eigenvalues = sorted(Cmat_eigenvalues, reverse=True) #sort

    Cmat_as_vec = []
    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if (j>=i):
                Cmat_as_vec += [Cmat[i,j]]

    pad_width = (padded_size**2 - padded_size)//2 + padded_size - ((num_atoms_file**2 - num_atoms_file)//2 + num_atoms_file)
    Cmat_as_vec = Cmat_as_vec + [0]*pad_width

    Cmat_as_vec = np.array(Cmat_as_vec)

    pad_width = padded_size - num_atoms_file
    Cmat_eigenvalues = np.pad(Cmat_eigenvalues, ((0, pad_width)), mode='constant')

    return Cmat_eigenvalues, Cmat_as_vec


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def coulombmat_and_eigenvalues_as_vec(filename, padded_size, sort=True):
    """
    returns Coulomb matrix and **sorted** Coulomb matrix eigenvalues
    Args:
        filename : (string) the .xyz input filename for the molecule
        padded_size : the number of atoms in the biggest molecule to be considered (same as padded eigenvalue vector length)
    Returns:
        (Eigenvalues vector, Coulomb matrix vector) as Numpy arrays
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

    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if i == j:
                Cmat[i,j]=0.5*chargearray[i]**2.4   # Diagonal terms
            else:
                dist=np.linalg.norm(xyzmatrix[i,:] - xyzmatrix[j,:])
                Cmat[i,j]=chargearray[i]*chargearray[j]/dist   #Pair-wise repulsion

    Cmat_eigenvalues = np.linalg.eigvals(Cmat)

    if (sort): Cmat_eigenvalues = sorted(Cmat_eigenvalues, reverse=True) #sort

    Cmat_as_vec = []
    for i in range(num_atoms_file):
        for j in range(num_atoms_file):
            if (j>=i):
                Cmat_as_vec += [Cmat[i,j]]

    pad_width = (padded_size**2 - padded_size)//2 + padded_size - ((num_atoms_file**2 - num_atoms_file)//2 + num_atoms_file)
    Cmat_as_vec = Cmat_as_vec + [0]*pad_width

    Cmat_as_vec = np.array(Cmat_as_vec)

    pad_width = padded_size - num_atoms_file
    Cmat_eigenvalues = np.pad(Cmat_eigenvalues, ((0, pad_width)), mode='constant')

    return Cmat_eigenvalues, Cmat_as_vec


#----------------------------------------------------------------------------
