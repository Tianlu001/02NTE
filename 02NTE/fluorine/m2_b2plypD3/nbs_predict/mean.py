import numpy as np
import sys

error= sys.argv[1]

#data=np.loadtxt(error)
#
#mean_error=np.mean(data)
#print(mean_error)


data=np.loadtxt(error, delimiter=',')

error1 = np.abs(data[:,-1])

mean_error1 = np.mean(error1)
print(mean_error1)


