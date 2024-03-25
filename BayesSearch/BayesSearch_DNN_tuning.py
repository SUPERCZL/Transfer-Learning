import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import optuna

from xekr_databuild import data

# Freezing a layer passes the layer name, freezing all layers above a layer passes self
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    
def load_model(model_path,hidden_dim1,hidden_dim2,hidden_dim3):
    Model = DNN(hidden_dim1,hidden_dim2,hidden_dim3).to(device=device)
    Model.load_state_dict(torch.load(model_path))
    return Model


def train(epoch, epochs, x_train, y_train,
          x_test, y_test, model,
          loss_fun, optimizer_data,
          list_1, list_2, list_3):
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
                     lr, epochs):
    global all_predit_list

    test_pre_fold = []
    test_MSE_fold = []
    test_RMSE_fold = []
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_verify_fold = x_train[test_index]
        y_verify_fold = y_train[test_index]

        x_train_fold = torch.from_numpy(np.array(x_train_fold)).to(device=device, dtype=torch.float32)
        y_train_fold = torch.from_numpy(np.array(y_train_fold)).to(device=device, dtype=torch.float32)
        x_verify_fold = torch.from_numpy(x_verify_fold).to(device=device, dtype=torch.float32)

        learningrate = lr

        Net = DNN(input_num, hidden_dim1, hidden_dim2, hidden_dim3).to(device=device)

        optimizer = torch.optim.Adam(Net.parameters(), lr=learningrate)
        loss_fun = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            train(epoch, epochs, x_train_fold, y_train_fold,
                  x_verify_fold, y_verify_fold,
                  Net, loss_fun, optimizer,
                  test_pre_fold, test_MSE_fold, test_RMSE_fold)

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
        self.hidden1 = nn.Linear(in_, hidden_dim1,bias=True)
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.hidden3 = nn.Linear(hidden_dim2, hidden_dim3)
        freeze(self)
        self.predict = nn.Linear(hidden_dim3, 1)
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        return output[:, 0]
    
def objective(trial):
    lr = trial.suggest_float('lr',0.00001,0.001,log=True) #整数型，(参数名称，下界，上界，步长)
    hidden_dim1 = 8
    hidden_dim2 = 8
    hidden_dim3 = 7
    epochs = trial.suggest_categorical('epochs',[15000, 20000, 25000])
    
    R2, MSE, RMSE = train_Kflod_data(x_train, y_train, 
                         hidden_dim1, hidden_dim2, hidden_dim3,
                         lr,epochs)
    
    return R2

if __name__ == '__main__':
    # build the database
    feature = 'Geometric+Energy+Chemical'
    target_ = 'Adsorption_Xe'#Selectivity
    database = 'MOF-2019-norm'

    x_train, y_train, type_ = \
        data(database, feature, target_, feature_list=True).data_in()

    x_test, y_test, type_ = \
        data('hCOF-norm', feature, target_, feature_list=True).data_in()

    input_num = len(type_)

    # split data (300 hCOF)
    SEED = 10
    x_train, y_train ,x_test, y_test = split_data(x_test, y_test, 300, SEED)
    
    # kfold
    SEED = 10
    NFOLDS = 5
    kf = KFold(n_splits = NFOLDS,random_state=SEED,shuffle=True)

    device = torch.device("cuda")

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=120)

    print('Best hyperparameters: ', study.best_params)
    print('Best score: ', study.best_value)    
    
    
    
        










