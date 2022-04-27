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

from feature_kojin import *
import pickle 
import joblib
import pybel

def load_data(): 
    x_train1 = x_train 
    y_train1 = y 
    return model_selection.train_test_split(x_train1, y_train1, test_size=0.10, random_state=256) 


def multi_RF(*data):
    x_train, x_test, y_train, y_test=data
    tuned_parameters = [{"n_estimators": np.linspace(5, 200, 50).astype('int')}]
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters)
    clf.fit(x_train, y_train)

    print("Best parameters set:", clf.best_params_)
    print("Optimized train:", clf.score(x_train, y_train))
    print("train error:", np.mean(((clf.predict(x_train)-y_train)**2)**0.5))
    print("Optimized Score:", clf.score(x_test, y_test))
    print("test error:", np.mean(((clf.predict(x_test)-y_test)**2)**0.5))

    
def test_KNN(*data):
    x_train, x_test, y_train, y_test=data
    regr = KNeighborsRegressor(n_neighbors=10,weights='distance')
    regr.fit(x_train, y_train)
    y_predict0 = regr.predict(x_train)
    
    
    print('Score: %6.2f' %regr.score(x_train, y_train))
    print('Residual sum of standard deviation: %.4f' %np.mean(((regr.predict(x_train)-y_train)**2)**0.5))
    
    print('Score: %6.2f' %regr.score(x_test, y_test))
    print('Residual sum of standard deviation: %.4f' %np.mean(((regr.predict(x_test)-y_test)**2)**0.5))


x_train, x_test, y_train, y_test=load_data()
regr = MLPRegressor(
    hidden_layer_sizes=(6, 2 ),  activation='relu', solver='adam', alpha=0.01, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000000, shuffle=True,
    random_state=1, tol=0.01, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)

regr.fit(x_train, y_train)

print('Score: %6.2f' %regr.score(x_train, y_train))
print('Residual sum of standard deviation: %.4f' %np.mean(((regr.predict(x_train)-y_train)**2)**0.5))

print('Score: %6.2f' %regr.score(x_test, y_test))
print('Residual sum of standard deviation: %.4f' %np.mean(((regr.predict(x_test)-y_test)**2)**0.5))


   

