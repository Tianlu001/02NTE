from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
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

x_train = np.array(x_train)
y_train = np.array(y_train)

def load_data(): 
    x_train1 = x_train 
    y_train1 = y 
    return model_selection.train_test_split(x_train1, y_train1, test_size=0.20, random_state=256) 


def test_Lasso(*data):
    x_train, x_test, y_train, y_test=data
    regr = linear_model.Lasso(alpha = 5.0e-2, normalize=True, max_iter=20000000)
    regr.fit(x_train, y_train)

    y_predict0  = regr.predict(x_train)
    y_predict10 = regr.predict(x_test)

    print(y_test[0], end=',     ')
    print(y_predict10[0], end=',     ')
    print(np.mean(y_predict10 - y_test))


def test_KRR(*data):
    x_train, x_test, y_train, y_test=data
    regr = KernelRidge(alpha=1e-9, kernel='rbf', gamma=1e-9)
    regr.fit(x_train, y_train)
    
    y_predict0  = regr.predict(x_train)
    y_predict10 = regr.predict(x_test)


    print(y_test[0], end=',     ')
    print(y_predict10[0], end=',     ')
    print(np.mean(y_predict10 - y_test))

    
def test_SVR(*data):
    x_train, x_test, y_train, y_test=data
    regr = SVR(C=8.37e+3, epsilon=1.75e0)
    regr.fit(x_train, y_train)
    
    y_predict0  = regr.predict(x_train)
    y_predict10 = regr.predict(x_test)


    print(y_test[0], end=',     ')
    print(y_predict10[0], end=',     ')
    print(np.mean(y_predict10 - y_test))


def test_BRR(*data):
    x_train, x_test, y_train, y_test=data
    regr = BayesianRidge(alpha_1=2.0e+2, alpha_2=1.0e-5, lambda_1=1.6e+5, lambda_2=1.4e+5)
    regr.fit(x_train, y_train)
    
    y_predict0  = regr.predict(x_train)
    y_predict10 = regr.predict(x_test)


    print(y_test[0], end=',     ')
    print(y_predict10[0], end=',     ')
    print(np.mean(y_predict10 - y_test))


def test_RFR(*data):
    x_train, x_test, y_train, y_test=data
    regr = RandomForestRegressor(n_estimators=70)
    regr.fit(x_train, y_train)
    
    y_predict0  = regr.predict(x_train)
    y_predict10 = regr.predict(x_test)


    print(y_test[0], end=',     ')
    print(y_predict10[0], end=',     ')
    print(np.mean(y_predict10 - y_test))


def test_KNN(*data):
    x_train, x_test, y_train, y_test=data
    regr = KNeighborsRegressor(n_neighbors=5)
    regr.fit(x_train, y_train)
    
    y_predict0  = regr.predict(x_train)
    y_predict10 = regr.predict(x_test)


    print(y_test[0], end=',     ')
    print(y_predict10[0], end=',     ')
    print(np.mean(y_predict10 - y_test))


#x_train, x_test, y_train, y_test=load_data()

loo = LeaveOneOut()
#n = loo.get_n_splits(x_train)
#print(n)

for train_index, test_index in loo.split(x_train):
    x_train1, x_test1 = x_train[train_index], x_train[test_index]
    y_train1, y_test1 = y_train[train_index], y_train[test_index]
   #test_Linear(x_train1, x_test1, y_train1, y_test1)
   #test_Lasso(x_train1, x_test1, y_train1, y_test1)
    test_KRR(x_train1, x_test1, y_train1, y_test1)
   #test_SVR(x_train1, x_test1, y_train1, y_test1)
   #test_BRR(x_train1, x_test1, y_train1, y_test1)
   #test_RFR(x_train1, x_test1, y_train1, y_test1)
   #test_KNN(x_train1, x_test1, y_train1, y_test1)



