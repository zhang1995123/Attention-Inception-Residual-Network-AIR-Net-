
from air_in import *
from loss import *
# from data_loader import *   ### your data_loader


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import TensorDataset, DataLoader

import os
import numpy as np
import pandas as pd
import time
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


file_path = 'D:/data_3D/'   ### your data_path
save_path = './result/'  ### your model_save_path
os.makedirs(save_path, exist_ok=True)   # make file_folder

epoch = 100
Batch_size = 36
best_loss = 1.
best_acc = 80.

# train_dataset = load_data(file_dir=file_path, val=val_num, test=test_num, mode='train') ### your data_loader
# val_dataset = load_data(file_dir=file_path, val=val_num, test=test_num, mode='val') ### your data_loader

model = inception_3(num_classes=1)
model = model.cuda()

### load dataset
train_loader = DataLoader(dataset = train_dataset, 
                        batch_size = Batch_size,
                        shuffle = True)
val_loader = DataLoader(dataset = val_dataset, 
                        batch_size = Batch_size,
                        shuffle = True)

### train & validation
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

t_start = time.time()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8, dampening=0, nesterov=True)
# loss_bce = nn.BCELoss()


for ep in range(epoch):
    len_data = len(train_loader)
    total_loss = 0.0

    lr = 0.0001 * (0.1 ** (ep // 30))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    gamma = 2
    if ep+1 >= 40:
        gamma = ep / 20.0

    ## train
    print('\n---train---')
    # print('lr =', lr, 'gamma =', gamma)
    total = 0.
    acc = 0.

    model.train()
    for i, (x, y) in enumerate(train_loader):
        total += x.size(0)
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        
        loss = focal_loss(out, y, alpha=0.5, gamma=gamma)

        out = F.sigmoid(out)
        
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        pred = (out >= 0.5).type(torch.FloatTensor)
        acc += torch.sum(pred.cpu().detach().eq(y.cpu().detach().view_as(pred))).item()

        # print statistics
        print('epoch:[%d/%d] iter:[%d/%d] loss: %.4f'
            % (ep+1, epoch, i+1, len_data, loss.item()))

    acc = acc / total * 100
    train_acc_list.append(acc)
    print(acc)
    
    total_loss /= len(train_loader)
    train_loss_list.append(total_loss)
    
    ## validation
    val_loss = 0.0
    total = 0
    acc = 0.0

    print('---eval---')
    with torch.no_grad():
        model.eval()
        for i, (x, y) in enumerate(val_loader):
            total += x.size(0)
            x, y = x.cuda(), y.cuda()
            pred = model(x)

            loss = focal_loss(pred, y, alpha=0.5, gamma=gamma)
            
            pred = F.sigmoid(pred)
            #loss = criterion(pred, torch.max(y,1)[1])  # y.max(dim=1)
            val_loss += loss.item()
            
            pred = (pred >= 0.5).type(torch.FloatTensor)
            #pred = torch.max(pred, 1)[1]
            acc += torch.sum(pred.cpu().detach().eq(y.cpu().detach().view_as(pred))).item()
            # #acc += np.sum(pred.cpu().detach().numpy() == y.view_as(pred).cpu().detach().numpy())

        # print(acc, '/', total)

        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        acc = acc / total * 100
        val_acc_list.append(acc)
        print('loss = ', val_loss)
        print('accuracy = ', acc, '%')

        # save model
        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path+'/model.pth')

        if acc >= best_acc:
            best_acc = acc
    
    print(label_name, val_num, ' current best :', best_acc, best_loss)

t_end = time.time()
print(t_end - t_start, 'sec')

name = ['train', 'val', 'train_acc', 'val_acc']
f = np.stack((train_loss_list, val_loss_list, train_acc_list, val_acc_list), 1)
log = pd.DataFrame(columns=name, data=f)
log.to_excel(save_path+'/model_curve'.xlsx')


print('------ done ------')
