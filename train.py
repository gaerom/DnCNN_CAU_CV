import import_ipynb
from model import DnCNN
from utils import MSE_loss
import utils as utils

from data_loader import DenoisingDataset
import data_loader as dl

import easydict
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

args = easydict.EasyDict({"model": 'DnCNN', "batch_size": 128, "train_data": 'data/train', 'sigma': 25, "epoch": 1, "lr": 1e-3, "optimizer":'Adam', "scheduler": 'MultiStepLR'})
# 모델, 배치 사이즈, 학습데이터, 노이즈 레벨, 에포크, 학습율, (옵티마이저, 스케쥴러)
# batch_size -> 64, 128 / sigma -> 10, 25, 50 / optimizer -> adam, sgd / num_layer -> 17 20 25

# cuda
cuda = torch.cuda.is_available()
# setting
batch_size = args.batch_size
n_epoch = args.epoch
sigma = args.sigma
optimizer = args.optimizer
scheduler = args.scheduler
save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# from dataloader_utils import train, train_dataloader, test_dataloader
def train_dataloader(train_dataset):
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        num_workers=2,
        persistent_workers=True,
        sampler=torch.utils.data.RandomSampler(train_dataset),
    )
    return train_dataloader

if __name__ == '__main__':
    print("Model Constuction")

    model = DnCNN()
    
    model.train()

    criterion = MSE_loss()

    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = utils.get_lr_scheduler(args.scheduler, optimizer=optimizer)

    # Measure time in PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    xs = dl.datagenerator(data_dir=args.train_data)
    xs = xs.astype('float32')/255.0
    xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
    DDataset = DenoisingDataset(xs, sigma)
    DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)

    for epoch in range(0, n_epoch):
        epoch_loss = 0

        # start time
        start.record()
        for n_count, xy in enumerate(DLoader):
            low_img, gt_img = xy[0].cuda(), xy[1].cuda() # low_img: noisy image, gt_img:ground truth
            optimizer.zero_grad()
            pred_img = model(low_img)
            loss = criterion(pred_img, gt_img)
            epoch_loss = loss.item()
            loss.backward()
            optimizer.step()
        # end time
        end.record()
        torch.cuda.synchronize()
        scheduler.step()
        print(f'epoch : {epoch+1}, loss : {epoch_loss/batch_size}, time : {start.elapsed_time(end)}')
    torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))