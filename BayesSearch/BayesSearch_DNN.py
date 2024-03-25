import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

import optuna

from xekr_databuild import data

def train(epoch,epochs,x_train,y_train,
          x_test,y_test,model,
          loss_fun,optimizer_data,
          list_1,list_2,list_3):

    model.train()
    train_predit = model(x_train)
    loss = loss_fun(train_predit, y_train)
    optimizer_data.zero_grad()
    loss.backward()
    optimizer_data.step()
    
    if epoch % epochs == 0:
        predit = model(x_test)
        predit = predit.detach().cpu().numpy()
        corr_test = round(r2_score(y_test, predit), 6)
        # MSE
        mse = np.sum((y_test - predit) ** 2) / len(y_test)
        # RMSE
        rmse = np.sqrt(mse)
        
        list_1.append(corr_test)
        list_2.append(mse)
        list_3.append(rmse)

def train_Kflod_data(x_train, y_train, input_num,
                     hidden_dim1, hidden_dim2, hidden_dim3,
                     lr,epochs):
    global all_predit_list
    
    test_pre_fold = []
    test_MSE_fold = []
    test_RMSE_fold = []
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_verify_fold = x_train[test_index]
        y_verify_fold = y_train[test_index]
        
        x_train_fold = torch.from_numpy(np.array(x_train_fold)).to(device=device,dtype = torch.float32)
        y_train_fold = torch.from_numpy(np.array(y_train_fold)).to(device=device,dtype = torch.float32)
        x_verify_fold = torch.from_numpy(x_verify_fold).to(device=device,dtype=torch.float32)

        learningrate = lr

        Net = DNN(input_num, hidden_dim1, hidden_dim2, hidden_dim3).to(device=device)

        optimizer = torch.optim.Adam(Net.parameters(), lr=learningrate)
        loss_fun = nn.MSELoss()

        for epoch in range(1,epochs+1):
            train(epoch, epochs, x_train_fold, y_train_fold,
                  x_verify_fold, y_verify_fold,
                  Net, loss_fun, optimizer,
                  test_pre_fold,test_MSE_fold,test_RMSE_fold)
            
    mean_R2 = np.mean(test_pre_fold)
    mean_MSE = np.mean(test_MSE_fold)
    mean_RMSE = np.mean(test_RMSE_fold)
    
    return mean_R2, mean_MSE, mean_RMSE

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
    
#build DNN
class DNN(nn.Module):
    def __init__(self, in_, hidden_dim1, hidden_dim2, hidden_dim3):
        super(DNN,self).__init__()
        self.hidden1 = nn.Linear(in_,hidden_dim1,bias=True)
        self.hidden2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.hidden3 = nn.Linear(hidden_dim2,hidden_dim3)
        self.predict = nn.Linear(hidden_dim3,1)
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        return output[:, 0]
    
def objective(trial):
    lr = trial.suggest_float('lr', 0.00001, 0.001, log=True)
    hidden_dim1 = trial.suggest_int('hidden_dim1', 2, 15)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 2, 15)
    hidden_dim3 = trial.suggest_int('hidden_dim3', 2, 15)
    epochs = trial.suggest_categorical('epochs', [10000, 15000, 20000, 40000])
    
    R2, MSE, RMSE = train_Kflod_data(x_train, y_train, input_num,
                         hidden_dim1, hidden_dim2, hidden_dim3,
                         lr,epochs)
    
    return R2

if __name__ == '__main__':
    # build the database
    feature = 'Geometric+Energy+Chemical'
    target_ = 'Adsorption_Xe'
    database = 'MOF-2019-norm'

    x_train, y_train, type_ = \
        data(database, feature, target_, feature_list=True).data_in()
    input_num = len(type_)
    
    # kfold
    SEED = 10
    NFOLDS = 5
    kf = KFold(n_splits = NFOLDS,random_state=SEED,shuffle=True)

    device = torch.device("cuda")
    
    # set optuna
    study = optuna.create_study(direction='maximize')
    
    # run
    study.optimize(objective, n_trials=100)

    print('feature')
    print('Best hyperparameters: ', study.best_params)
    print('Best score: ', study.best_value)    
    
    
    
        








