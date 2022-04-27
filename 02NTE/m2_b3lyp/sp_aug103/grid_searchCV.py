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



#y_train = y_train[:,0]


def load_data(): 
    x_train1 = x_train 
    y_train1 = y_train
    return model_selection.train_test_split(x_train1, y_train1, test_size=0.20, random_state=452) 

def multi_Lasso(*data):
    x_train, x_test, y_train, y_test=data
    tuned_parameters = [{"alpha": np.logspace(-9, 1, 100)}]
    clf = GridSearchCV(Lasso(max_iter = 200000000, normalize=True), tuned_parameters)
    clf.fit(x_train, y_train)

    
    print("Best parameters set:", clf.best_params_)
    print("train error:", np.mean(((clf.predict(x_train) - y_train)**2)**0.5))
    print("test error:", np.mean(((clf.predict(x_test) - y_test)**2)**0.5))


def multi_KRR(*data):
    x_train, x_test, y_train, y_test=data
    tuned_parameters = [ {"alpha": np.logspace(-15, -5, 20), "gamma": np.logspace(-12, -5, 20), "kernel" : ['rbf']}]
    clf = GridSearchCV(KernelRidge(), tuned_parameters)
    clf.fit(x_train, y_train)

    
    print("Best parameters set:", clf.best_params_)
    print("train error:", np.mean(((clf.predict(x_train) - y_train)**2)**0.5))
    print("test error:", np.mean(((clf.predict(x_test) - y_test)**2)**0.5))


def multi_SVR(*data):
    x_train, x_test, y_train, y_test=data
    tuned_parameters = [{"C": np.logspace(-2, 5, 40), "epsilon": np.logspace(-3, 3, 40)}]
    clf = GridSearchCV(SVR(), tuned_parameters)
    clf.fit(x_train, y_train)

    
    print("Best parameters set:", clf.best_params_)
    print("train error:", np.mean(((clf.predict(x_train) - y_train)**2)**0.5))
    print("test error:", np.mean(((clf.predict(x_test) - y_test)**2)**0.5))


def multi_BRR(*data):
    x_train, x_test, y_train, y_test=data
    tuned_parameters = [{"alpha_1": np.logspace(-3, 2, 20),"alpha_2": np.logspace(-5,1,20), "lambda_1": np.logspace(1,6,20),"lambda_2": np.logspace(2,8,20)}]
    clf = GridSearchCV(BayesianRidge(), tuned_parameters)
    clf.fit(x_train, y_train)


    print("Best parameters set:", clf.best_params_)
    print("train error:", np.mean(((clf.predict(x_train) - y_train)**2)**0.5))
    print("test error:", np.mean(((clf.predict(x_test) - y_test)**2)**0.5))


def multi_RF(*data):
    x_train, x_test, y_train, y_test=data
    tuned_parameters = [{"n_estimators": np.linspace(1, 100, 100).astype('int')}]
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters)
    clf.fit(x_train, y_train)

    print("Best parameters set:", clf.best_params_)
    print("train error:", np.mean(((clf.predict(x_train) - y_train)**2)**0.5))
    print("test error:", np.mean(((clf.predict(x_test) - y_test)**2)**0.5))


def multi_KNN(*data):
    x_train, x_test, y_train, y_test=data
    tuned_parameters = [{"n_neighbors": np.linspace(1,20,19).astype('int')}]
    clf = GridSearchCV(KNeighborsRegressor(), tuned_parameters)
    clf.fit(x_train, y_train)

    print("Best parameters set:", clf.best_params_)
    print("train error:", np.mean(((clf.predict(x_train) - y_train)**2)**0.5))
    print("test error:", np.mean(((clf.predict(x_test) - y_test)**2)**0.5))


x_train, x_test, y_train, y_test=load_data()
#multi_Lasso(x_train, x_test, y_train, y_test)
multi_KRR(x_train, x_test, y_train, y_test)
#multi_SVR(x_train, x_test, y_train, y_test)
#multi_RF(x_train, x_test, y_train, y_test)
#multi_BRR(x_train, x_test, y_train, y_test)
#multi_KNN(x_train, x_test, y_train, y_test)

   

