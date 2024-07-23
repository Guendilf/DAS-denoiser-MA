import os
import sys
import time
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

import config

from models.N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self, U_Net_origi, U_Net, TestNet
from metric import Metric
from masks import Mask
from utils import *
from transformations import *

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from absl import app
from torch.utils.tensorboard import SummaryWriter

num_mc = 1
sigma_n = 1
max_epochs = 1

def get_lr_lambda(initial_lr, step_size, lr_decrement):
    def lr_lambda(step):
        #return max(0.0, initial_lr - (step // step_size) * lr_decrement / initial_lr)
        exp = step / step_size
        exp = int(exp) #staircase
        return lr_decrement ** exp #*initial_lr TODO: muss ich nicht nochh die multiplikation machhen
    return lr_lambda

def n2same(noise_images, device, model, lambda_inv=2):
    mask, marked_points = Mask.mask_random(noise_images, maskamount=0.005, mask_size=(1,1))
    mask = mask.to(device)
    masked_input = (1-mask) * noise_images #delete masked pixels in noise_img
    masked_input = masked_input + (torch.normal(0, 0.2, size=noise_images.shape).to(device) * mask ) #deleted pixels will be gausian noise with sigma=0.2 as in appendix D
    denoised = model(noise_images)
    denoised_mask = model(masked_input)
    mse = torch.nn.MSELoss()
    loss_rec = torch.mean((denoised-noise_images)**2) # mse(denoised, noise_images)
    loss_inv = torch.sum(mask*(denoised-denoised_mask)**2)# mse(denoised, denoised_mask)
    loss = loss_rec + lambda_inv *sigma_n* (loss_inv/marked_points).sqrt()
    return loss, denoised, loss_rec, loss_inv, marked_points


def train(model, device, optimizer, scheduler, dataLoader, mode, writer, rauschen, lambda_inv, epoch, option=""):
    losses = []
    psnr = []
    lex = 0
    lin = 0
    all_marked = 0
    n = torch.empty((0, 3*128*128)).to(device)
    for batch_idx, (original, label) in enumerate(tqdm(dataLoader)):
        if "komplex" in rauschen:
            original = original*255
            poison = 255.0 * np.random.poisson(30.0*(original/255.0))/30.0
            gauss =  np.random.normal(0, 60, original.shape)
            bernoulli_noise_map = np.random.binomial(1, 0.5, original.shape)*255
            bernoulli_noised = np.where(np.random.uniform(0, 1, original.shape)<0.2, bernoulli_noise_map, gauss+poison)
            noise = np.clip(bernoulli_noised, 0, 255.0)
            noise_image = (torch.from_numpy(noise)/255.0).to(device).type(torch.float32)
        else:
            noise = (torch.rand_like(original*255.0)*0.4).to(device).type(torch.float32)
            noise_image = torch.clip(original*255.0 + noise, 0,255.0)/255.0
        original = original.to(device)
        if "train" in mode:
            loss, denoised, _, _, _ = n2same(noise_image, device, model, lambda_inv)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            writer.add_scalar('Train loss', loss, epoch * len(dataLoader) + batch_idx)
            writer.add_scalar('Train psnr', Metric.calculate_psnr(original, denoised).item(), epoch * len(dataLoader) + batch_idx)
        elif "val" in mode:
            with torch.no_grad():
                loss, denoised, loss_rec, loss_inv, marked_pixel = n2same(noise_image, device, model, lambda_inv)
                if "estimate" in option:
                    all_marked += marked_pixel
                    lex += loss_rec
                    lin += loss_inv
                    n_partition = (denoised-noise_image).view(denoised.shape[0], -1) # (b, c*w*h)
                    n_partition = torch.sort(n_partition, dim=1).values
                    n = torch.cat((n, n_partition), dim=0)
                    if batch_idx == len(dataLoader)-1:
                        e_l = 0
                        for i in range(num_mc): #kmc
                            samples = torch.tensor(torch.multinomial(n.view(-1), n.shape[1], replacement=True))#.view(1, n.shape[1])
                            samples = torch.sort(samples).values
                            e_l += torch.mean((n-samples)**2)
                        lex = lex / (len(dataLoader) * denoised.shape[0])
                        lin = lin / all_marked
                        e_l = e_l / num_mc
                        estimated_sigma = (lin)**0.5 + (lin + lex-e_l)**0.5
                        print('new sigma_loss is ', estimated_sigma)
                        if estimated_sigma < sigma_n:
                            sigma_n = float(estimated_sigma)
                            print('sigma_loss updated to ', estimated_sigma)
                        writer.writer.add_scalar('estimated sigma', estimated_sigma, epoch)
            writer.add_scalar('Val loss', loss, epoch * len(dataLoader) + batch_idx)
            writer.add_scalar('Val psnr', Metric.calculate_psnr(original, denoised).item(), epoch * len(dataLoader) + batch_idx)
        else:
            with torch.no_grad():
                denoised = model(noise_image)
            writer.add_scalar('Test psnr', Metric.calculate_psnr(original, denoised).item(), epoch * len(dataLoader) + batch_idx)
            loss = 0
        psnr.append(Metric.calculate_psnr(original, denoised))
        losses.append(loss)

    return loss, psnr
        


          

def main(argv):
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"


    celeba_dir = config.celeba_dir
    methodeListe = ['n2same', 'n2info']
    end_results = pd.DataFrame(columns=methodeListe)

    
    transform_noise = transforms.Compose([
        transforms.CenterCrop((128,128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        #transforms.Lambda(lambda x:  x * 2 -1),
        ])
    print("lade Datensätze ...")
    dataset = datasets.CelebA(root=celeba_dir, split='train', download=False, transform=transform_noise)
    dataset = torch.utils.data.Subset(dataset, list(range(6400)))
    
    dataset_validate = datasets.CelebA(root=celeba_dir, split='valid', download=False, transform=transform_noise)
    dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(640)))

    dataset_test = datasets.CelebA(root=celeba_dir, split='test', download=False, transform=transform_noise)
    dataset_test = torch.utils.data.Subset(dataset_test, list(range(640)))

    store_path = log_files()
    for methode in methodeListe:
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
        if torch.cuda.device_count() == 1:
            model = N2N_Orig_Unet(3,3).to(device) #default
        else:
            model = U_Net(first_out_chanel=96, batchNorm=True).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.methodes[methode]['lr'])
        lr_lambda = get_lr_lambda(config.methodes[methode]['lr'], config.methodes[methode]['changeLR_steps'], config.methodes[methode]['changeLR_rate'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        writer = SummaryWriter(log_dir=os.path.join(store_path ,"tensorboard"))

        for epoch in range(max_epochs):
            loss, psnr = train(model, device, optimizer, scheduler, dataLoader, 'train', writer, rauschen='komplex', lambda_inv=2, epoch=epoch, option="")

            loss, psnr = train(model, device, optimizer, scheduler, dataLoader_validate, 'val', writer, rauschen='komplex', lambda_inv=2, epoch=epoch, option="")
        
        loss, psnr = train(model, device, optimizer, scheduler, dataLoader_test, 'test', writer, rauschen='komplex', lambda_inv=2, epoch=0, option="")




if __name__ == "__main__":
    app.run(main)