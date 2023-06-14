# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:15:26 2023
@author: Shaina Mehta
"""

from __future__ import annotations
__author__: list[str] = ['Shaina Mehta', 'Anjali Gaur', 'Prof. (Dr.) M. Partha Sarathi']

__license__: str = r'''
    MIT License
    Copyright (c) 2023 Shaina Mehta
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''


__doc__: str = r'''
    >>> Paper Title: 
            A Simplified Method of Detection and Predicting the Severity of Knee Osteoarthritis
    
    >>> Project Abstract: 
            Knee osteoarthritis (OA) is a degenerative joint disease caused by the gradual wear and tear of 
            the knee's articulatory cartilage. It is most typically found in adults. It is a progressive 
            illness that can cause impairment if it is not treated or diagnosed in its early stages [1]. 
            This paper presents a simplified approach for detecting and predicting knee osteoarthritis based 
            on the Kellgren-Lawrence (KL) grading system using Residual Networks. The baseline of the X-Ray 
            images was obtained from Mendeley Data [2]. The following four approaches are proposed for grading 
            the severity of Knee Osteoarthritis: the first distinguishes severity according to the original 
            KL Grading System, the second distinguishes normal (KL 0-I) from osteoarthritic knees (KL II-IV), 
            the third distinguishes severity as normal (KL 0-I), non-severe (KL II), or severe (KL III-IV), 
            and the fourth distinguishes severity level from KL Grade 0 to KL Grade III. The empirical results 
            showed that the performance improved with fewer multiclass labels being used, with binary class 
            labels and three class labels surpassing all the other models proposed in this paper. The model for 
            binary class labels achieved a 90% accuracy rate and weighted Precision, Recall, and F1 scores of 
            90%, 90%, and 90%, respectively whereas the model for three classes labels achieved a 90% accuracy 
            rate and weighted Precision, Recall, and F1 scores of 90%, 90%, and 89%, respectively. The proposed
            models also showed how they might be used to classify patients early in the course of the disease, 
            slowing its progression and raising their quality of life.
        
    >>> Keywords:
            Knee Osteoarthritis (KOA), Deep Learning, X-Ray Images, Kellgren -Lawrence Grading System, 
            Convolutional Neural Networks (CNN), ResNet18, MobileNetV2.
            
    >>> Acknowledgement: 
            The satisfaction that accompanies the successful completion of any task would be incomplete without 
            the mention of people whose ceaseless cooperation made it possible, whose constant guidance and 
            encouragement crowns all the efforts with success. I would like to thank my supervisor, 
            Dr M. Partha Sarathi, Professor, Department of Computer Science and Engineering at Amity University,
            Noida as well as my parents and my grandmother Mr Dheeraj Mehta, Ms Shikha Mehta, and 
            Ms Kiran Mehta who are the biggest driving force behind my successful completion of the project. 
            They have always been there to solve any query of mine and guided me in the right direction 
            regarding the project. Without their help and inspiration, I would not have been able to complete 
            my project. I would also like to thank my project partner Ms Anjali Gaur as well as the batchmates 
            and juniors Ms Leah Khan, Mr Rahul Sawhney, Mr Nikhil J. Dutta, Mr Rakshit Walia, Mr Aadil Sehrawat,
            Mr Venkatesh, Ms Arushi Kumar, Ms Ayushi Pandit, Ms Deepansha Adlakha, Mr Amartya Sumukh Varma, 
            Ms Vanshika Gupta, Ms Harjot Kaur, Ms Prerna Singh, and Ms Tanya Yadav for constantly motivating 
            me and giving ideas during the project whenever I got stuck. I would like to thank a few faculty 
            members of the Computer Science and Engineering Department, Amity University, Noida - Dr Vibha Nehra, 
            Dr Ram Paul, Dr Rajni Sehgal, and Dr S. K. Dubey who guided me, helped me, and gave me ideas and 
            motivation at each step.
'''
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import torch.optim as optim
from torch.optim import lr_scheduler as lrs
from typing import ClassVar, Optional
import copy
import seaborn as sns
import math
from skimage import morphology
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from torchvision import models
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive/Knee Osteoarthritis/"

class MetaData:
    def collectData(self, path: str) -> pd.DataFrame:
        grades = ['0/','1/','2/','3/']
        finalData = []
        finalLabels = []
        for i in range(len(grades)):
            dir_join = path+grades[i]
            for file in os.listdir(dir_join):
                finalData.append(dir_join+file)
                finalLabels.append(i)
        dat = {'file_name':finalData,'label':finalLabels}
        return pd.DataFrame.from_dict(dat)
    
    def splitData(self) -> tuple:
        final_data = self.collectData(path)
        X = final_data.iloc[:,0].values
        Y = final_data.iloc[:,1].values
        x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
        train_dict = {'file_name':x_train,'label':y_train}
        test_dict = {'file_name':x_test,'label':y_test}
        train_data = pd.DataFrame.from_dict(train_dict)
        test_data = pd.DataFrame.from_dict(test_dict)
        return (train_data, test_data) 
    
    def underSample(self, df: pd.DataFrame, max_samples: int, min_samples:int, column: str) -> pd.DataFrame:
      df=df.copy()
      groups=df.groupby(column)    
      trimmed_df = pd.DataFrame(columns = df.columns)
      groups=df.groupby(column)
      for label in df[column].unique(): 
        group=groups.get_group(label)
        count=len(group)    
        if count > max_samples:
          sampled_group=group.sample(n=max_samples, random_state=123,axis=0)
          trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)
        else:
          if count>=min_samples:
            sampled_group=group        
            trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)
      trimmed_df = trimmed_df.sample(frac = 1)
      trimmed_df = trimmed_df.reset_index(drop=True)
      trimmed_df['label'] = pd.to_numeric(trimmed_df['label'])
      print('after trimming, the maximum samples in any class is now ',max_samples, ' and the minimum samples in any class is ', min_samples)
      return trimmed_df
    

class KneeXrayData(Dataset):
  def __init__(self, data: pd.DataFrame, transform: A.transforms) -> None:
    self.data=data
    self.transform=transform

  def __len__(self) -> int:
    return len(self.data)
  
  def __getitem__(self, idx: int) -> tuple:
    path = self.data.iloc[idx]['file_name']
    label = torch.tensor(int(self.data.iloc[idx]['label']))  
    image = cv2.imread(path,cv2.COLOR_BGR2GRAY)
    if self.transform is not None:
      image = self.transform(image=image)['image']
    return (image, label)
class AnalyzeData:
  def dataDistribition(self, data: pd.DataFrame, type: str) -> 'plot':
    df = pd.DataFrame()
    df['ostheoarthritis grading'] = ['Grade 0','Grade 1','Grade 2','Grade 3']
    grade_count = []
    for i in range(4):
      c = len(data[data['label']==i])
      grade_count.append(c)
    df['count ('+ type + ')'] = grade_count[:]
    print('Table for Osteoarthitis Data Distribition for',type)
    print(df)
    print()
    x = list(np.arange(4))
    y = list(df['count ('+ type + ')'])
    plt.bar(x,y,color=['#9a01ff','#04b8ff', '#01ffd5','#b7ff01'])
    plt.grid()
    plt.xticks(x,list(df['ostheoarthritis grading']))
    plt.xlabel('ostheoarthritis grading')
    plt.ylabel('count ('+ type + ')')
    plt.title('Knee Ostheoarthritis')
    plt.show()

  def showImages(self,ImgLoader: torch.utils.data.dataloader.DataLoader, title: str) -> 'plot':
    fig = plt.figure(figsize = (14, 7))
    plt.title(title)
    for inputs, _ in ImgLoader:
      for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, xticks = [], yticks = [])     
        plt.imshow(inputs[i].numpy().transpose(1, 2, 0),cmap='gray')
      break

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False
        if np.isnan(metrics):
            return True
        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            print('improvement!')
        else:
            self.num_bad_epochs += 1
            print(f'no improvement, bad_epochs counter: {self.num_bad_epochs}')

        if self.num_bad_epochs >= self.patience:
            return True
        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class ModelUtils:
  def allDevice(self) -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

  def trainEpoch(self, model: 'model', dataloader: torch.utils.data.dataloader.DataLoader, lr: float, 
                 optimizer: object , loss_fn: object, device: torch.device) -> tuple:
    total_loss,total_acc,count = 0.0,0.0,0
    model.train()
    for x,y in dataloader:
      #print(type(x))
      x = x.to(device)
      y = y.to(device)
      optimizer.zero_grad()
      outputs = model(x)
      _, preds = torch.max(outputs , 1)
      loss = loss_fn(outputs,y)
      loss.backward()
      optimizer.step()
      total_acc+=(preds==y).sum()
      total_loss+=loss
      count+=len(y)
    return (total_loss.item()/count, total_acc.item()/count)
  
  def validateEpoch(self, model: 'model', dataloader: torch.utils.data.dataloader.DataLoader, 
                    loss_fn: object, device: torch.device) -> tuple: 
    total_loss,total_acc,count = 0.0,0.0,0
    model.eval()
    with torch.no_grad():
      for x,y in dataloader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        _, preds = torch.max(outputs , 1)
        loss = loss_fn(outputs,y)
        total_acc+=(preds==y).sum()
        total_loss+=loss
        count+=len(y)
    return (total_loss.item()/count, total_acc.item()/count)
  
  def train(self, model: 'model', train_dataloader: torch.utils.data.dataloader.DataLoader, 
            val_dataloader: torch.utils.data.dataloader.DataLoader, 
            lr: float, optimizer: object, loss_fn: object, epochs: int, device: torch.device) -> tuple:
    es = EarlyStopping(patience=3)
    terminate_training = False
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    #res: dict[str, list[float]] = { 'train_loss' : [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for ep in range(epochs):
      tl,ta = self.trainEpoch(model,train_dataloader,lr,optimizer,loss_fn, device)
      vl,va = self.validateEpoch(model,val_dataloader,loss_fn, device)
      print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
      #res['train_loss'].append(tl)
      #res['train_acc'].append(ta)
      #res['val_loss'].append(vl)
      #res['val_acc'].append(va)
      #scheduler.step()
      if va > best_acc:
        best_acc = va
        best_model_wts = copy.deepcopy(model.state_dict())
      if es.step(vl):
          terminate_training = True
          print('early stop criterion is met, we can stop now')
          break
    return (best_acc,best_model_wts)

  def plot_results(self, hist: dict[str, list[float]]) -> 'plot':
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')
    plt.plot(hist['val_acc'], label='Validation acc')
    plt.legend()
    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')
    plt.plot(hist['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

  def testing(self, model: 'model', dataloader: torch.utils.data.dataloader.DataLoader, 
              device: torch.device) -> tuple:
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
      for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_test_pred = model(x_batch)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())
    #print(y_true_list[0])
    y_pred_list = [i[0] for i in y_pred_list]
    y_true_list = [i[0] for i in y_true_list]
    return (y_pred_list,y_true_list)

 class Utils:
  def confusion_matrix(self, model, actual, preds, device):
    cfm = confusion_matrix(actual, preds)
    class_names = ['Grade 0','Grade 1','Grade 2','Grade 3']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks,class_names)
    plt.yticks(tick_marks,class_names)
    sns.heatmap(pd.DataFrame(cfm),annot=True,cmap='CMRmap_r')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.Text(0.5,257.44,'Predicted label')
    plt.show()
  
  def classification_report(self, model, actual, preds, device):
    print(classification_report(actual, preds, digits=4))
  
  
## Driver Code
  
if __name__ == "__main__":
    train_data, test_data = MetaData().splitData()
    max_samples=1157 # since each class has more than 200 images all classes will be trimmed to have 200 images per class
    min_samples=734
    column='label'
    train_df=  MetaData().underSample(train_data, max_samples, min_samples, column)

    # Show Distribution of Data

    print('Showing Class Distribution')

    print('For Training Set')
    AnalyzeData().dataDistribition(train_data, 'Training Set')

    print('For Test Set')
    AnalyzeData().dataDistribition(test_data, 'Test Set')

    print('For Balanced Train Set')
    AnalyzeData().dataDistribition(train_df, 'Balanced Train Set')

    # Train and Validation Transform

    train_transform = A.Compose([
        A.Resize(width=299, height=299),
        A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, p=1.0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(9,9), p=1.0),
        A.CenterCrop(width=280, height=200),
        A.Resize(width=299, height=299),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.Normalize(mean=(0.4671), std=(0.2907)),
        ToTensorV2()
    ])

    # Test Transform

    test_transform = A.Compose([
        A.Resize(width=299, height=299),
        A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, p=1.0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(9,9), p=1.0),
        A.CenterCrop(width=280, height=200),
        A.Resize(width=299, height=299),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.Normalize(mean=(0.6085), std=(0.1542)),
        ToTensorV2()
    ])

    # Setting Hyper-parameters
    
    num_classes = 10
    num_epochs =20
    batch_size = 50
    learning_rate = 0.001
    
    # Allocating the GPU and Loading the Model in GPU

    device = ModelUtils().allDevice()
    model = models.mobilenet_v2(pretrained=True)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=4, bias=True)
    model.to(device)

    # Setting up Optimizers, Learning Rate Scheduler and Loss Function

    optimizer= optim.Adam(model.parameters(),lr = 0.01)
    #scheduler = lrs.StepLR(optimizer,step_size =7,gamma=0.1,verbose=True)
    loss_fn = nn.CrossEntropyLoss()
    kf = StratifiedKFold(n_splits=10, shuffle=False)
    acc = []
    wts = []

    x = train_df.iloc[:,0].values
    y = train_df.iloc[:,1].values

    index = 0
    for train_idx, val_idx in kf.split(x,y):
        index += 1
        print(index, 'fold')
        x_t, x_v = x[train_idx], x[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        td = {'file_name':x_t,'label':y_t}
        vd = {'file_name':x_v,'label':y_v}
        t_data = pd.DataFrame.from_dict(td)
        v_data = pd.DataFrame.from_dict(vd)
        train_image_dataset = KneeXrayData(data=t_data, transform=train_transform)
        val_image_dataset = KneeXrayData(data=v_data, transform=train_transform)
        train_image_loader = DataLoader(train_image_dataset, batch_size = 50, shuffle = False)
        val_image_loader = DataLoader(val_image_dataset, batch_size = 50, shuffle = False)
        best_acc, best_model = ModelUtils().train(model,train_image_loader,val_image_loader,0.0001,optimizer,loss_fn,100,device)
        acc.append(best_acc)
        wts.append(best_model)
        model.load_state_dict(best_model)    

    for i in range(10):
        bm = wts[i]
        print('Model Version'+str(i+1))
        torch.save(bm,'/content/drive/My Drive/Model Version'+str(i+1)+'.pth')
        m = models.mobilenet_v2(pretrained=True)
        m.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        m.classifier[1] = nn.Linear(in_features=1280, out_features=4, bias=True)
        m.load_state_dict(torch.load('/content/drive/My Drive/Model Version'+str(i+1)+'.pth',map_location=device))
        m.to(device)
        m.eval()
        train_image_dataset = KneeXrayData(data=train_df, transform=train_transform)
        train_image_loader = DataLoader(train_image_dataset, batch_size = 50, shuffle = False)
        test_image_dataset = KneeXrayData(data=test_data, transform=test_transform)
        test_image_loader = DataLoader(test_image_dataset, batch_size = 50, shuffle = False)
        # Plotting The Confusion Matrix and Classification Report of Train Data
        preds, actual = ModelUtils().testing(m,train_image_loader,device) 
        # Confusion Matrix
        Utils().confusion_matrix(m,actual,preds,device)
        # Classification Report 
        Utils().classification_report(m,actual,preds,device)  
        # Plotting The Confusion Matrix and Classification Report of Test Data
        preds, actual = ModelUtils().testing(m,test_image_loader,device)
        # Confusion Matrix
        Utils().confusion_matrix(m,actual,preds,device)
        # Classification Report 
        Utils().classification_report(m,actual,preds,device)

