from model import DnCNN
from utils import MSE_loss
import utils as utils

from data_loader import DenoisingDataset
import data_loader as dl

import easydict
import json
import wandb
import os
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import ToPILImage

from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import KFold # cross-validation

# 실험 셋팅
expname = '[name]_exp[num]'
filename = expname+'.json'
args = easydict.EasyDict({"model": 'DnCNN', "batch_size": 128, "train_data": 'data/train', 'sigma': 25, "epoch": 1, "lr": 1e-3, "optimizer":'Adam', "scheduler": 'MultiStepLR'})
with open(filename,'w', encoding='utf-8') as f:
    json.dump(args, f, indent='\t')
# 모델, 배치 사이즈, 학습데이터, 노이즈 레벨, 에포크, 학습율, (옵티마이저, 스케쥴러)
# batch_size -> 64, 128 / sigma -> 10, 25, 50 / optimizer -> adam, sgd / num_layer -> 17 20 25

# wandb (wandb.login 필요)
wandb.init(
    # set the wandb project where this run will be logged
    project="dncnn",
    name=expname,

    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": args.model,
    "epochs": args.epoch,
    "batch_size" : args.batch_size,
    "sigma" : args.sigma,
    "num_layer" : args.num_layer,
    "optimizer" : args.optimizer,
    "lr_scheduler" : args.scheduler
    }
)

# cuda
cuda = torch.cuda.is_available()
# setting
k = 3 # k_fold cross validation
batch_size = args.batch_size
n_epoch = args.epoch
sigma = args.sigma
num_layer = args.num_layer
optimizer = args.optimizer
scheduler = args.scheduler
save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# from dataloader_utils import train, train_dataloader, test_dataloader
def train_dataloader(train_dataset):
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=True,
        sampler=torch.utils.data.RandomSampler(train_dataset),
    )
    return train_dataloader

if __name__ == '__main__':
    model = DnCNN()
    model.train()
    criterion = MSE_loss()

    xs = dl.datagenerator(data_dir=args.train_data)
    xs = xs.astype('float32')/255.0
    xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))
    DDataset = DenoisingDataset(xs, sigma)
    DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_indices, val_indices) in enumerate(kfold.split(DDataset)):
        print(f"fold: {fold+1}")

        train_dataset = torch.utils.data.Subset(DDataset, train_indices)
        val_dataset = torch.utils.data.Subset(DDataset, val_indices)

        train_loader = DataLoader(dataset=train_dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)

        model = DnCNN(num_layers=num_layer)

        if cuda:
            model = model.cuda()
        if optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = utils.get_lr_scheduler(args.scheduler, optimizer=optimizer)

        # Measure time in PyTorch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    
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
            # test
            test_gt_img = np.array(imread('/content/drive/MyDrive/DnCNN/data/Set12/01.png'), dtype=np.float32)/255.0
            np.random.seed(seed=0)
            noisy = test_gt_img + np.random.normal(0, sigma/255.0, test_gt_img.shape)  # Add Gaussian noise without clipping
            noisy = noisy.astype(np.float32)
            tensor_noisy = torch.from_numpy(noisy).view(1, -1, noisy.shape[0], noisy.shape[1])
            tensor_noisy = tensor_noisy.cuda()
            pred_img = model(tensor_noisy)  # inference
            pred_img = pred_img.view(noisy.shape[0], noisy.shape[1])
            test_pred_img = pred_img.cpu().detach().numpy().astype(np.float32)

            psnr_score = psnr(test_gt_img, test_pred_img)
            ssim_score = ssim(test_gt_img.squeeze(), test_pred_img.squeeze())
            wandb.log({'loss' : epoch_loss/batch_size,'time' : start.elapsed_time(end), 'image':wandb.Image(test_pred_img),'test_psnr':psnr_score, 'test_ssim':ssim_score})
            print(f'epoch : {epoch+1}, loss : {epoch_loss/batch_size}, time : {start.elapsed_time(end)}')
        torch.save(model, os.path.join(save_dir, f'model_{epoch+1}_{fold+1}fold.pth'))