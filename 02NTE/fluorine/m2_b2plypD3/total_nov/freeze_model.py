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


def train_Lasso(*data):
    x_train, y_train=data
    regr = linear_model.Lasso(alpha = 1.2e-3, normalize=True, max_iter=20000000)
    regr.fit(x_train, y_train)
    y_predict0 = regr.predict(x_train)
    
    print(np.mean(((y_predict0 - y_train)**2)**0.5))
    joblib.dump(regr, 'Lasso.joblib')
    


def train_KRR(*data):
    x_train, y_train=data
    regr = KernelRidge(alpha=2e-6, kernel='rbf', gamma=2e-6)
    regr.fit(x_train, y_train)
    y_predict0 = regr.predict(x_train)
    
    print(np.mean(((y_predict0 - y_train)**2)**0.5))
    joblib.dump(regr, 'KRR.joblib')
    

    
def train_SVR(*data):
    x_train, y_train=data
    regr = SVR(C=954, epsilon=5.87e0)
    regr.fit(x_train, y_train)
    y_predict  = regr.predict(x_train)
    
    print(np.mean(((y_predict - y_train)**2)**0.5))
    
    


def train_MLP(*data):
    x_train, y_train=data
    regr = MLPRegressor(hidden_layer_sizes=(40,40),activation='relu', solver='adam', alpha=0.01,max_iter=30000)
    regr.fit(x_train, y_train)
    y_predict  = regr.predict(x_train)
    
    print(np.mean(((y_predict - y_train)**2)**0.5))
    

def train_BR(*data):
    x_train, y_train=data
    regr = BayesianRidge(alpha_1=5e-0, alpha_2=1e-5, lambda_1=1e5, lambda_2=2e5 )
    regr.fit(x_train, y_train)
    y_predict   = regr.predict(x_train)
    
    print(np.mean(((y_predict - y_train)**2)**0.5))



def train_KNN(*data):
    x_train, y_train=data
    regr = KNeighborsRegressor(n_neighbors=5)
    regr.fit(x_train, y_train)
    y_predict   = regr.predict(x_train)

    print(np.mean(((y_predict - y_train)**2)**0.5))


def train_RF(*data):
    x_train, y_train=data
    regr = RandomForestRegressor(n_estimators=22)
    regr.fit(x_train, y_train)
    y_predict   = regr.predict(x_train)

    print(np.mean(((y_predict - y_train)**2)**0.5))


#x_train, x_test, y_train, y_test=load_data()

#train_Lasso(x_train, y_train)
train_KRR(x_train, y_train)
#train_SVR(x_train, y_train)
#train_MLP(x_train, y_train)
#train_BR(x_train, y_train)
#train_RF(x_train, y_train)
#train_KNN(x_train, y_train)

   

