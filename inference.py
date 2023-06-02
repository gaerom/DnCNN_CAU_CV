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

    # 가우시안 노이즈 추가
    np.random.seed(seed=1024)
    noisy_img = test_img + np.random.normal(0, sigma/255.0, test_img.shape)
    noisy_img = noisy_img.astype(np.float32)
    torch_nimg = torch.from_numpy(noisy_img).view(1, -1, noisy_img.shape[0], noisy_img.shape[1])

    torch.cuda.synchronize()
    start_time = time.time()
    y_ = y_.cuda()
    x_ = model(y_)  # inference
    x_ = x_.view(y.shape[0], y.shape[1])
    x_ = x_.cpu()
    x_ = x_.detach().numpy().astype(np.float32)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))



    


    # Define dataset_type based on the configuration
    dataset_type = opt['datasets']['train']['dataset_type']

    # Iterate over phases (train, test)
    for phase, dataset_opt in opt['datasets'].items():
        
        avg_psnr = 0.0
        test_dataset = define_Dataset(dataset_opt)
        test_dataloader = test_dataloader(test_dataset)

        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
            


    val_dataloader = test_dataloader
