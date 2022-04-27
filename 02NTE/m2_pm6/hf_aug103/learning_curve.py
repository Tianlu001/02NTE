from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, FastICA
from sklearn import datasets, linear_model, discriminant_analysis, model_selection

from heattt import *
import pickle 
import joblib
import pybel

def load_data(trainsize): 
    x_train1 = x_train 
    y_train1 = y_target
    return model_selection.train_test_split(x_train1, y_train1, test_size=40, train_size=trainsize, random_state=742) #54 409 1

x_validate = x_validation
y_validate = y2_target

def train_Lasso(*data):
    x_train, y_train=data
    regr = linear_model.Lasso(alpha = 1e-3, normalize=True, max_iter=20000000)
    regr.fit(x_train, y_train[:,0])
    y_predict0 = regr.predict(x_train)
    y_predict1 =  1.66 * y_train[:,2] / y_predict0
    
   #print('Score: %6.2f' %regr.score(x_train, y_train))
   #print('Train error: %.4f' %np.mean(((regr.predict(x_train)-y_train)**2)**0.5))
    print(np.mean(((y_predict1 - y_train[:,1])**2)**0.5))
    


def test_Lasso(*data):
    x_train, x_test, y_train, y_test=data
    regr = linear_model.Lasso(alpha = 7.0e-4, normalize=True, max_iter=20000000)
    regr.fit(x_train, y_train[:,0])


    y_predict0  = regr.predict(x_train)
    y_predict1  =  1.66 * y_train[:,2] / y_predict0

    y_predict10 = regr.predict(x_test)
    y_predict11 =  1.66 * y_test[:,2] / y_predict10

    y_predict100 = regr.predict(x_validate)
    y_predict101 =  1.66 * y_validate[:,2] / y_predict100
    

    print(np.mean(((y_predict1 - y_train[:,1])**2)**0.5),end=' ')
    print(np.mean(((y_predict11 - y_test[:,1])**2)**0.5),end=' ')
    print(np.mean(((y_predict101 - y_validate[:,1])**2)**0.5))
   #print(np.mean(((y_predict11 - y_test[:,1])**2)**0.5))


def train_KRR(*data):
    x_train, y_train=data
    regr = KernelRidge(alpha=2e-4, kernel='rbf', gamma=9e-7)
    regr.fit(x_train, y_train[:,0])

    y_predict0 = regr.predict(x_train)
    y_predict1 =  1.66 * y_train[:,2] / y_predict0
    
    print(np.mean(((y_predict1 - y_train[:,1])**2)**0.5))
    
    

def test_KRR(*data):
    x_train, x_test, y_train, y_test=data
    regr = KernelRidge(alpha=5e-14, kernel='rbf',gamma=5e-16)
    regr.fit(x_train, y_train)
    
    y_predict0  = regr.predict(x_train)
    y_predict10 = regr.predict(x_test)
    y_predict100 = regr.predict(x_validate)

    print(np.mean(((y_predict0 - y_train)**2)**0.5),end=' ')
    print(np.mean(((y_predict10 - y_test)**2)**0.5),end=' ')
    print(np.mean(((y_predict100 - y_validate)**2)**0.5))

    
def train_SVR(*data):
    x_train, y_train=data
    regr = SVR(C=471, epsilon=5.0e-5)
    regr.fit(x_train, y_train[:,0])
    y_predict0 = regr.predict(x_train)
    y_predict1 =  1.66 * y_train[:,2] / y_predict0
    
    print(np.mean(((y_predict1 - y_train[:,1])**2)**0.5))
    
    
    
def test_SVR(*data):
    x_train, x_test, y_train, y_test=data
    regr = SVR(C=471, epsilon=5.0e-5)
    regr.fit(x_train, y_train[:,0])

    y_predict0  = regr.predict(x_train)
    y_predict1  =  1.66 * y_train[:,2] / y_predict0
    y_predict10 = regr.predict(x_test)
    y_predict11 =  1.66 * y_test[:,2] / y_predict10
    

   #print("train error:", np.mean(((y_predict1 - y_train[:,1])**2)**0.5))
   #print("test error:", np.mean(((y_predict11 - y_test[:,1])**2)**0.5))
    print(np.mean(((y_predict11 - y_test[:,1])**2)**0.5))

for i in range(20, 301, 2):    
    print(i, end=' ')
    x_train3, x_test3, y_train3, y_test3=load_data(i)
   #test_Lasso(x_train3, x_test3, y_train3, y_test3)
    test_KRR(x_train3, x_test3, y_train3, y_test3)

   

