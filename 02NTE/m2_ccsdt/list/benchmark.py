import numpy as np
import pandas as pd
import sys

DFT = sys.argv[1]
TrueHgas = 'Hnzpe_pwpb95.csv'

csv_data0 = pd.read_csv(TrueHgas) 
data0 = np.array(csv_data0)

#print(data0[:,-1])

csv_data1 = pd.read_csv(DFT) 
data1 = np.array(csv_data1)

#print(data1[:,-1])


error= data1[:,-1]-data0[:,-1]
error1 = np.abs(error)
print(error.max())
print(np.where(error==np.max(error)))
mean_error1 = np.mean(error1)
print(mean_error1)


