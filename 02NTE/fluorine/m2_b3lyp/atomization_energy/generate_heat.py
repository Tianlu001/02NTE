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


x_train = X_Atype_list
print(x_train)

y_train = y1

def load_data():
    x_train = X_Cmat_eigs
    y_train = y
    return model_selection.train_test_split(x_train, y_train, test_size=0.70, random_state=1)


#x_train, x_test, y_train, y_test=load_data()

#regr = linear_model.Lasso(alpha = 0.015, max_iter=200000)
regr = LinearRegression(fit_intercept=False)
regr.fit(x_train, y_train)

print(regr.intercept_)
print(regr.coef_)
#parameters = np.transpose([summed_BoH_feature_names,regr.coef_])
#print(parameters)

y_predict0 = regr.predict(x_train)


print('Score: %6.2f' %regr.score(x_train, y_train))
print('Residual sum of standard deviation: %.4f' %np.mean(((regr.predict(x_train)-y_train)**2)**0.5))


