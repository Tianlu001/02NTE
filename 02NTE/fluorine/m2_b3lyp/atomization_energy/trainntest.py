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

#from feature import *
from heat import *
import pickle 
import joblib
import pybel

#from heat22 import *
#x_validate = x_validation
#y_validate = y_validation

def load_data(): 
    x_train1 = x_train 
    y_train1 = y_train
    return model_selection.train_test_split(x_train1, y_train1, test_size=0.20, random_state=7503) 



def train_Lasso(*data):
    x_train, y_train=data
    regr = linear_model.Lasso(alpha = 3e1, normalize=True, max_iter=20000000)
    regr.fit(x_train, y_train)
    y_predict  = regr.predict(x_train)
    
   #print('Score: %6.2f' %regr.score(x_train, y_train))
   #print('Train error: %.4f' %np.mean(((regr.predict(x_train)-y_train)**2)**0.5))
    print(np.mean(((y_predict - y_train)**2)**0.5))
    


def test_Lasso(*data):
    x_train, x_test, y_train, y_test=data
    regr = linear_model.Lasso(alpha = 3e1, normalize=True, max_iter=20000000)
    regr.fit(x_train, y_train)
    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)
    

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
   #print(np.mean(((y_predict11 - y_test[:,1])**2)**0.5))


def train_KRR(*data):
    x_train, y_train=data
    regr = KernelRidge(alpha=5e-4, kernel='rbf',gamma=5e-7)
    regr.fit(x_train, y_train)
    y_predict = regr.predict(x_train)
    
    print(np.mean(((y_predict - y_train)**2)**0.5))
    
    

def test_KRR(*data):
    x_train, x_test, y_train, y_test=data
    regr = KernelRidge(alpha=5e-4, kernel='rbf',gamma=5e-7)
    regr.fit(x_train, y_train)


    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
   #y_predict100 = regr.predict(x_validate)
   #print("validate error:", np.mean(((y_predict100 - y_validate)**2)**0.5))

    
def train_SVR(*data):
    x_train, y_train=data
    regr = SVR(C=954, epsilon=5.87e0)
    regr.fit(x_train, y_train)
    y_predict  = regr.predict(x_train)
    
    print(np.mean(((y_predict - y_train)**2)**0.5))
    
    
    
def test_SVR(*data):
    x_train, x_test, y_train, y_test=data
    regr = SVR(C=1854, epsilon=5.87e0)
    regr.fit(x_train, y_train)

    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)
    

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
   #y_predict100 = regr.predict(x_validate)
   #print("validate error:", np.mean(((y_predict100 - y_validate)**2)**0.5))


    
x_train, x_test, y_train, y_test=load_data()
train_Lasso(x_train, y_train)
test_Lasso(x_train, x_test, y_train, y_test)
#train_KRR(x_train, y_train)
#test_KRR(x_train, x_test, y_train, y_test)
#train_SVR(x_train, y_train)
#test_SVR(x_train, x_test, y_train, y_test)

   

