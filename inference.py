import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from models import DnCNN
from utils import MSE_loss, get_lr_scheduler
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave


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

def inference(opt):
    # Config 파일에서 가져오기
    PATH = './models/'
    MODEL_PATH = PATH+'DnCNN.pth'
    TEST_IMG_PATH = ''
    sigma = 25
    OUTPUT_PATH = './results/'

    model = DnCNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model.cuda()

    test_img = np.array(imread(os.path.join(TEST_IMG_PATH)))
    # timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 가우시안 노이즈 추가
    np.random.seed(seed=1024)
    noisy_img = test_img + np.random.normal(0, sigma/255.0, test_img.shape)
    noisy_img = noisy_img.astype(np.float32)
    torch_nimg = torch.from_numpy(noisy_img).view(1, -1, noisy_img.shape[0], noisy_img.shape[1])

    start.record()
    torch_nimg = torch_nimg.cuda()
    inference_img = model(torch_nimg)
    inference_img = inference_img.view(noisy_img.shape[0], noisy_img.shape[1])
    inference_img = inference_img.cpu()
    inference_img = inference_img.detach().numpy().astype(np.float32)
    end.record()
    torch.cuda.synchronize()

    psnr_score = compare_psnr(test_img, inference_img)
    ssim_score = compare_ssim(test_img, inference_img)
    # save img
    imsave(OUTPUT_PATH, np.clip(inference_img,0,1))
    print(f'{TEST_IMG_PATH} Image saved to {OUTPUT_PATH}, psnr : {psnr_score}, ssim : {ssim_score}, time : {start.elapsed_time(end)}')
