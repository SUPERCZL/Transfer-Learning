import numpy as np
import random
import optuna

from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# import the two-stage algorithm
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2

from xekr_databuild import data
import joblib

def split_data(x, y, num_train, seed):
    index_array = list(range(len(x)))
    random.seed(seed)
    index_number = random.sample(index_array, num_train)
    x_train = x[index_number]
    y_train = y[index_number]
    for i in index_number:
        index_array.remove(i)
    x_test = x[index_array]
    y_test = y[index_array]
    return x_train, y_train, x_test, y_test

def out_val(y_,y_pre):
    mse = mean_squared_error(y_pre, y_)
    R2 = r2_score(y_, y_pre)
    return mse, R2

def train_trada_XGB(N, steps, fold, random_state, sample_size):
    global filename, R2_aoc, save_
    regr_1 = TwoStageTrAdaBoostR2(XGBR(n_estimators = 100,
                                       max_depth = 15,
                                       learning_rate = 0.08,
                                       tree_method="gpu_hist",
                                       gpu_id = 0),
                                  n_estimators=N, sample_size=sample_size,
                                  steps=steps, fold=fold,
                                  random_state=random_state)

    regr_1.fit(x_train, y_train)
    y_pred1 = regr_1.predict(x_test)
    mse_train, R2_train = out_val(y_train_, regr_1.predict(x_train_))
    mse_test, R2_test = out_val(y_test, y_pred1)
    print('XGB')
    print('Train R2:', round(R2_train, 5), '\nTrain MSE:', mse_train)
    print('Test R2:', round(R2_test, 5), '\nTest MSE:', mse_test)
    if R2_test > R2_aoc and save_:
        print('R2_aoc', R2_aoc)
        joblib.dump(regr_1, filename)
        R2_aoc = R2_test
    return R2_test

def train_trada_RF(N, steps, fold, random_state, sample_size):
    global filename, R2_aoc, save_
    regr_1 = TwoStageTrAdaBoostR2(RandomForestRegressor(n_estimators=35,
                                                        max_depth=12,
                                                        min_samples_split=7,
                                                        min_samples_leaf=7,
                                                        n_jobs=-1,
                                                        random_state=10),
                                  n_estimators=N, sample_size=sample_size,
                                  steps=steps, fold=fold,
                                  random_state=random_state)

    regr_1.fit(x_train, y_train)
    y_pred1 = regr_1.predict(x_test)
    mse_train, R2_train = out_val(y_train_, regr_1.predict(x_train_))
    mse_test, R2_test = out_val(y_test, y_pred1)
    print('RF')
    print('Train R2:', round(R2_train, 5), '\nTrain MSE:', mse_train)
    print('Test R2:', round(R2_test, 5), '\nTest MSE:', mse_test)
    if R2_test > R2_aoc and save_:
        print('R2_aoc', R2_aoc)
        joblib.dump(regr_1, filename)
        R2_aoc = R2_test
    return R2_test

def train_trada_GBRT(N, steps, fold, random_state, sample_size):
    global filename, R2_aoc, save_
    regr_1 = TwoStageTrAdaBoostR2(GradientBoostingRegressor(
                                       learning_rate=0.07201345603761897
                                     , n_estimators=80
                                     , min_samples_split=12
                                     , min_samples_leaf=12
                                     , max_depth=6
                                     , init=None, max_features=None
                                     , alpha=0.9, verbose=0, max_leaf_nodes=None
                                     , warm_start=False
                                     , random_state=10),
                                  n_estimators=N, sample_size=sample_size,
                                  steps=steps, fold=fold,
                                  random_state=random_state)

    regr_1.fit(x_train, y_train)
    y_pred1 = regr_1.predict(x_test)
    mse_train, R2_train = out_val(y_train_, regr_1.predict(x_train_))
    mse_test, R2_test = out_val(y_test, y_pred1)
    print('GBRT')
    print('Train R2:', round(R2_train, 5), '\nTrain MSE:', mse_train)
    print('Test R2:', round(R2_test, 5), '\nTest MSE:', mse_test)
    if R2_test > R2_aoc and save_:
        print('R2_aoc', R2_aoc)
        joblib.dump(regr_1, filename)
        R2_aoc = R2_test
    return R2_test

def objective_XGB(trial):
    n_estimators = trial.suggest_int('n_estimators', 5, 20)
    steps = trial.suggest_int('steps', 2, 10)
    random_state = np.random.RandomState(1)

    R2 = train_trada_XGB(n_estimators, steps, 5, random_state, sample_size)
    return R2

def objective_RF(trial):
    n_estimators = trial.suggest_int('n_estimators', 5, 20)
    steps = trial.suggest_int('steps', 2, 10)
    random_state = np.random.RandomState(1)

    R2 = train_trada_RF(n_estimators, steps, 5, random_state, sample_size)

    return R2

def objective_GBRT(trial):
    n_estimators = trial.suggest_int('n_estimators', 5, 20)
    steps = trial.suggest_int('steps', 2, 10)
    random_state = np.random.RandomState(1)

    R2 = train_trada_GBRT(n_estimators, steps, 5, random_state, sample_size)

    return R2


if __name__ == '__main__':
    feature = 'Geometric+Energy+Chemical'
    traget_ = 'Adsorption_Xe'
    TL_material = 'hCOF'

    x_train, y_train, type_ = \
        data('MOF-2019', feature, feature_list=True, traget_=traget_).data_in()

    x_test, y_test, type_ = \
        data(database_name_=TL_material, feature_=feature, feature_list=True, traget_=traget_).data_in()

    n_source_train = len(x_train)
    n_target_train = 300
    n_target_test = len(x_test) - 300

    x_train_, y_train_, x_test, y_test = split_data(x_test, y_test, n_target_train, 10)
    x_train = np.vstack((x_train, x_train_))
    y_train = np.hstack((y_train, y_train_))
    sample_size = [n_source_train, n_target_train]

    # save the best model
    save_ = True

    model_type = 'XGB'
    model_ = '/model_finish/'
    filename = model_+'Tradaboost_' + TL_material + '_' + feature + '_' + model_type + '.sav'
    R2_aoc = 0

   # optuna
    study = optuna.create_study(direction='maximize')
    if model_type == 'RF':
        study.optimize(objective_RF, n_trials=10)
    elif model_type == 'GBRT':
        study.optimize(objective_GBRT, n_trials=10)
    elif model_type == 'XGB':
        study.optimize(objective_XGB, n_trials=10)

    print('TL_'+model_type+' of '+TL_material+' : '+feature)
    print('Best hyperparameters: ', study.best_params)
    print('Best score: ', study.best_value)
