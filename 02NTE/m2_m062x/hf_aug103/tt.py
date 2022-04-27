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

from heat import *
import pickle 
import joblib
import pybel

def load_data(): 
    x_train1 = x_train 
    y_train1 = y_train
    return model_selection.train_test_split(x_train1, y_train1, test_size=20,train_size=60, random_state=471) 


def train_Lasso(*data):
    x_train, y_train=data
    regr = linear_model.Lasso(alpha = 1e-3, normalize=True, max_iter=20000000)
    regr.fit(x_train, y_train)
    y_predict0 = regr.predict(x_train)
    
   #print('Score: %6.2f' %regr.score(x_train, y_train))
   #print('Train error: %.4f' %np.mean(((regr.predict(x_train)-y_train)**2)**0.5))
    print(np.mean(((y_predict0 - y_train)**2)**0.5))
    


def test_Lasso(*data):
    x_train, x_test, y_train, y_test=data
    regr = linear_model.Lasso(alpha = 1e-2, normalize=True, max_iter=20000000)
    regr.fit(x_train, y_train)
    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
   #print(np.mean(((y_predict11 - y_test[:,1])**2)**0.5))


def train_KRR(*data):
    x_train, y_train=data
    regr = KernelRidge(alpha=4e-5, kernel='rbf',gamma=8e-7)
    regr.fit(x_train, y_train)
    y_predict0 = regr.predict(x_train)
    
    print(np.mean(((y_predict0 - y_train)**2)**0.5))
    
    

def test_KRR(*data):
    x_train, x_test, y_train, y_test=data
    regr = KernelRidge(alpha=5e-6, kernel='rbf',gamma=5e-6)
    regr.fit(x_train, y_train)

    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)
    y_predict100 = regr.predict(x_validate)

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
    print("validate error:", np.mean(((y_predict100 - y_validate)**2)**0.5))

    
def train_SVR(*data):
    x_train, y_train=data
    regr = SVR(C=954, epsilon=5.87e0)
    regr.fit(x_train, y_train)
    y_predict  = regr.predict(x_train)
    
    print(np.mean(((y_predict - y_train)**2)**0.5))
    
    
    
def test_SVR(*data):
    x_train, x_test, y_train, y_test=data
    regr = SVR(C=1455, epsilon=8.87e0)
    regr.fit(x_train, y_train)

    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)
    
    y_predict100 = regr.predict(x_validate)

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
    print("validate error:", np.mean(((y_predict100 - y_validate)**2)**0.5))


def train_MLP(*data):
    x_train, y_train=data
    regr = MLPRegressor(hidden_layer_sizes=(40,40),activation='relu', solver='adam', alpha=0.01,max_iter=30000)
    regr.fit(x_train, y_train)
    y_predict  = regr.predict(x_train)
    
    print(np.mean(((y_predict - y_train)**2)**0.5))
    
def test_MLP(*data):
    x_train, x_test, y_train, y_test=data
    regr = MLPRegressor(hidden_layer_sizes=(50,40),activation='relu', solver='adam', alpha=0.02,max_iter=30000)
    regr.fit(x_train, y_train)

    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)
    
    y_predict100 = regr.predict(x_validate)

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
    print("validate error:", np.mean(((y_predict100 - y_validate)**2)**0.5))


def test_BR(*data):
    x_train, x_test, y_train, y_test=data
    regr = BayesianRidge(alpha_1=5e-0, alpha_2=1e-5, lambda_1=1e5, lambda_2=2e5 )
    regr.fit(x_train, y_train)


    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)
    
    y_predict100 = regr.predict(x_validate)

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
    print("validate error:", np.mean(((y_predict100 - y_validate)**2)**0.5))


def test_KNN(*data):
    x_train, x_test, y_train, y_test=data
    regr = KNeighborsRegressor(n_neighbors=5)
    regr.fit(x_train, y_train)


    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)
    
    y_predict100 = regr.predict(x_validate)

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
    print("validate error:", np.mean(((y_predict100 - y_validate)**2)**0.5))


def test_RF(*data):
    x_train, x_test, y_train, y_test=data
    regr = RandomForestRegressor(n_estimators=22)
    regr.fit(x_train, y_train)


    y_predict0  = regr.predict(x_train)
    y_predict1  = regr.predict(x_test)
    
    y_predict100 = regr.predict(x_validate)

    print("train error:", np.mean(((y_predict0 - y_train)**2)**0.5))
    print("test  error:", np.mean(((y_predict1 - y_test)**2)**0.5))
    print("validate error:", np.mean(((y_predict100 - y_validate)**2)**0.5))

x_train, x_test, y_train, y_test=load_data()
#train_Lasso(x_train, y_train)
test_Lasso(x_train, x_test, y_train, y_test)
#train_KRR(x_train, y_train)
#test_KRR(x_train, x_test, y_train, y_test)
#train_SVR(x_train, y_train)
#test_SVR(x_train, x_test, y_train, y_test)
#train_MLP(x_train, y_train)
#test_MLP(x_train, x_test, y_train, y_test)
#test_BR(x_train, x_test, y_train, y_test)
#test_RF(x_train, x_test, y_train, y_test)
#test_KNN(x_train, x_test, y_train, y_test)

   

