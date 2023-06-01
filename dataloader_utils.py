import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from models import DnCNN

class MSE_loss(nn.modules.loss._Loss):
    def __init__(self):
        super(MSE_loss, self).__init__()
    def forward(self, y, target):
        return f.mse_loss(y, target)

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

def test_dataloader(test_dataset):
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=torch.utils.data.SequentialSampler(test_dataset),
    )
    return test_dataloader

def train(opt):
    lr = 0.01
    n_epoch = 100
    batch_size = 64
    PATH = './models/'

    # Define dataset_type based on the configuration
    dataset_type = opt['datasets']['train']['dataset_type']

    # Iterate over phases (train, test)
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_dataset = define_Dataset(dataset_opt)
            train_dataloader = train_dataloader(train_dataset)
        elif phase == 'test':
            test_dataset = define_Dataset(dataset_opt)
            test_dataloader = test_dataloader(test_dataset)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

        if phase == 'train':
            model = DnCNN()
            model.cuda()
            model.train()
            criterion = MSE_loss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for epoch in range(0, n_epoch):
                epoch_loss = 0
                st = time.time()
                for xy in train_dataloader:
                    x, y = xy[0].cuda(), xy[1].cuda()
                    optimizer.zero_grad()
                    prediction = model(x)
                    loss = criterion(prediction, y)
                    epoch_loss = loss.item()
                    loss.backward()
                    optimizer.step()
                et = time.time() - st
                print(f'epoch : {epoch}, loss : {epoch_loss/batch_size}, time : {et}')
            torch.save(model, PATH+'DnCNN.pth')

    val_dataloader = test_dataloader
