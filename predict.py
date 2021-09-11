from air_in import *
from loss import *
# from data_loader import *   ### your data_loader

import sys, os
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


file_path = 'D:/data_3D/'  ### your data_path
model_path = './result/model.pth'  ### your model_save_path

# test_dataset = load_data(file_dir=file_path,test=test_num, mode='test')  ### your data_loader
test_loader = DataLoader(dataset = test_dataset, batch_size=1, shuffle=False)

model = inception_3(num_classes=1)
model.load_state_dict(torch.load(model_path))
model = model.cuda()

total = 0
correct = 0
TP = 0
FP = 0
TN = 0
FN = 0

model.eval()
with torch.no_grad():
    for i, (data, label) in enumerate(test_loader):
        total += data.shape[0]
        data = data.cuda()
        label = label.item()

        pred = model(data)
        pred = torch.sigmoid(pred)

        pred = (pred >= 0.5).type(torch.FloatTensor)
        pred = pred.item()

        print(pred)
        

        if pred == label:
            correct += 1
        if pred == 1.0 and label == 1.0:
            TP += 1
        if pred == 1.0 and label == 0.0:
            FP += 1
        if pred == 0.0 and label == 0.0:
            TN += 1
        if pred == 0.0 and label == 1.0:
            FN += 1

        
    acc = correct / total
    print(model_path)
    print('accuracy =', acc)
    print('sensitivity', TP/(TP+FN))
    print('specificity', TN/(FP+TN))
    print('TP =', TP)
    print('TN =', TN)
    print('FP =', FP)
    print('FN =', FN)
