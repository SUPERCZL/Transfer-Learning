'''

Bayesian search for model parameter

'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor as XGBR
from skopt import BayesSearchCV
from skopt.space import Real, Integer
#from skopt import gp_minimize

from xekr_databuild import data



if __name__ == '__main__':
    # build the database
    feature = 'Geometric+Energy+Chemical'
    target_ = 'Adsorption_Xe'
    database = 'MOF-2019'
    
    x_train, y_train, type_ = \
    data(database, feature, target_, feature_list=True).data_in()


    # set params
    param_xgb = {
        'n_estimators': Integer(10, 100),
        'max_depth': Integer(2, 15),
        'learning_rate': Real(0.08, 0.23)
    }

    param_rf = {
        'n_estimators': Integer(10, 120),
        'max_depth': Integer(2, 15),
        'min_samples_split': Integer(2, 13),
        'min_samples_leaf': Integer(2, 13)
    }

    param_gbrt = {
        'n_estimators': Integer(20, 120),
        'max_depth': Integer(3, 12),
        'learning_rate': Real(0.05, 0.21),
        'min_samples_split': Integer(2, 12),
        'min_samples_leaf': Integer(2, 12),
    }

    import warnings
    warnings.filterwarnings("ignore")
    print('Geometric+Energy+Chemical')


    clf1 = XGBR()
    B1 = BayesSearchCV(clf1, param_xgb, n_iter=100, cv=5,
                          verbose=0, n_jobs=-1, refit=True)
    B1.fit(x_train, y_train)
    best_estimator = B1.best_params_
    print('\nXGB', best_estimator, '\n')
    print(B1.best_score_)

    
    clf2 = RandomForestRegressor()
    B2 = BayesSearchCV(clf2, param_rf, n_iter=100, cv=5,
                          verbose=0, n_jobs=-1, refit=True)
    B2.fit(x_train, y_train)
    best_estimator = B2.best_params_
    print('\nRF', best_estimator, '\n')
    print(B2.best_score_)

    
    clf3 = GradientBoostingRegressor()
    B3 = BayesSearchCV(clf3, param_gbrt, n_iter=100, cv=5,
                          verbose=0, n_jobs=-1, refit=True)
    B3.fit(x_train, y_train)
    best_estimator = B3.best_params_
    print('\nGBRT', best_estimator, '\n')
    print(B3.best_score_)
    