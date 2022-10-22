import pandas as pd 
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torchvision
from copy import deepcopy






class TrainDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.max_classes = 70
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, idx):
        image_name, labels = self.data_df.iloc[idx]['img'], self.data_df.iloc[idx].values[-8:],
        
        y = np.zeros(self.max_classes)
        for l in labels:
            if l != 0:
              y[l-1] = 1.0

        image = cv2.imread(f"./train/{image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(y).float()
    
    def __len__(self):
        return len(self.data_df)





class TestDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.max_classes = 70 + 1
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.data_df.iloc[idx]['img']

        image = cv2.imread(f"./test/{image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def __len__(self):
        return len(self.data_df)




data_df = pd.read_csv("./train.csv")



train_df, valid_df = train_test_split(data_df, test_size=0.00, random_state=0) # обучим для получения конечной модели на всем датасете

train_dataset = TrainDataset(train_df, train_transform)
valid_dataset = TrainDataset(valid_df, valid_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           pin_memory=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=1,
                                           pin_memory=True)



def train(res_model, criterion, optimizer, train_dataloader, test_dataloader, NUM_EPOCH=15):
    train_loss_log = []
    val_loss_log = []
    
    train_acc_log = []
    val_acc_log = []
    
    weighted_penalty = get_weight_classes()
    
    for epoch in tqdm(range(NUM_EPOCH)):
        model.train()
        train_loss = 0.
        train_size = 0
        max_val_acc = 0
        
        train_pred = []

        for imgs, labels in train_dataloader:
            optimizer.zero_grad()

            imgs = imgs.cuda()
            labels = labels.cuda()
            y_pred = model(imgs)

            loss = criterion(y_pred, labels)
            loss.backward()
            
            train_loss += loss.item()
            train_size += y_pred.size(0)
            train_loss_log.append(loss.data / y_pred.size(0))
            
            y_pred = np.where(y_pred.detach().cpu().numpy() > 0.5, 1, 0).astype(int).reshape(-1)
            labels = labels.detach().cpu().numpy().astype(int).reshape(-1)

            train_pred.append(np.sum(np.where(y_pred == labels, 1, 0)) / labels.shape[0])

            optimizer.step()

        train_acc_log.append(train_pred)

        val_loss = 0.
        val_size = 0
        
        val_pred = []
        
        model.eval()
        
        with torch.no_grad():
            for imgs, labels in test_dataloader:
                
                imgs = imgs.cuda()
                labels = labels.cuda()
                
                pred = model(imgs)
                loss = torch.nn.functional.binary_cross_entropy(pred, labels)
                
                val_loss += loss.item()
                val_size += pred.size(0)
                

                pred = np.where(pred.detach().cpu().numpy() > 0.5, 1, 0).astype(int).reshape(-1)
                labels = labels.detach().cpu().numpy().astype(int).reshape(-1)

                val_pred.append(np.sum(np.where(pred == labels, 1, 0)) / labels.shape[0])
        
        if np.mean(val_pred) > max_val_acc:
                max_val_acc = np.mean(val_pred)
                best_model = deepcopy(model)

        val_loss_log.append(val_loss / val_size)
        val_acc_log.append(val_pred)

        clear_output()
        


        print('Train loss:', (train_loss / train_size))
        print('Val loss:', (val_loss / val_size)*100)
        print('Train acc:', np.mean(train_pred)*100, '%')
        print('Val acc:', np.mean(val_pred)*100, '%')
        
    return train_loss_log, train_acc_log, val_loss_log, val_acc_log, best_model



class MyModel (nn.Module):
    def __init__(self, n_classes, model = None):
        super().__init__()
        if model is None:
            self.model = torchvision.models.resnet50(weights = 'IMAGENET1K_V2')

        else:
            self.model = model
        
        self.out= nn.Sequential(nn.Dropout(p=0.2),
                                nn.BatchNorm1d(1000),
                                nn.Linear(1000, n_classes),
                                nn.Sigmoid()
                               )
        
    def forward(self,x):
        x=self.model(x)
        out=self.out(x)
        return out



model = MyModel(70, None)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loss_log, train_acc_log, val_loss_log, val_acc_log, best_model = train(model, 
                                                                 criterion, 
                                                                 optimizer, 
                                                                 train_loader, 
                                                                 valid_loader, 
                                                                 100)



model.eval()

data_test_df = pd.read_csv("./test.csv")
test_dataset = TestDataset(data_test_df, valid_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=1,
                                           shuffle=False)

                                           
submission_df = pd.read_csv("./sample_solution.csv")

count = 0
t = 0.95
for imgs in tqdm(test_loader):
    
    imgs = imgs.cuda()
    pred = model(imgs)

    pred = pred.cpu().detach().numpy().reshape(-1)
    indexes = np.argsort(pred)[::-1] # сортируем по достоверности

    answer = np.zeros(8) # ответ в виде вектора из 8 чисел
    for i, idx in enumerate(indexes):
        if i>=8 or pred[idx] < t: 
            break
        answer[i] = idx+1
        
        sing_name = 'sing' + str(1 + i)
        submission_df.iloc[count][sing_name] = idx+1

    count+=1



submission_df.to_csv('./submission.csv', index=False)
