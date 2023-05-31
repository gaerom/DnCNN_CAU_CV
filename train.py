import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time

from models import DnCNN

# Loss function
class MSE_loss(nn.modules.loss._Loss):
    def __init__(self):
        super(MSE_loss, self).__init__()
    def forward(self, y, target):
        return f.mse_loss(y, target)

# Hyper Parameter
lr = 0.01
n_epoch = 100
batch_size = 64
PATH = './models/'

# train
def train():
    model = DnCNN()
    model.cuda()
    model.train()
    criterion = MSE_loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    for epoch in range(0, n_epoch):
        epoch_loss = 0
        # dataloader
        st = time.time()
        for xy in '''DLoader''' :
            x, y = xy[0].cuda(), xy[1].cuda()
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            epoch_loss = loss.item()
            loss.backward()
            optimizer.step()
        et = time.time() - st
        print(f'epoch : {epoch}, loss : {epoch_loss/batch_size} time : {et}')
    torch.save(model,PATH+'DnCNN.pth')
