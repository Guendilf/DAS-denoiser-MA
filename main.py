import math
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

from models.N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self
from metric import Metric
from masks import Mask
from utils import show_logs, show_pictures_from_dataset, log_files, show_tensor_as_picture,  normalize_image
from transformations import *

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid

#sys.path.insert(0, str(Path(__file__).resolve().parents[3])) #damit die Pfade auf dem Server richtig waren (copy past von PG)
from absl import app
from torch.utils.tensorboard import SummaryWriter
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\Code\DAS-denoiser-MA\runs"
#import wandb


max_Iteration = 10
max_Epochs = 2


transform = transforms.Compose([
    #transforms.Resize((128, 128)), #TODO:crop und dann resize
    transforms.RandomResizedCrop((128,128)),
    transforms.ToTensor(),                  # PIL-Bild in Tensor
    transforms.Lambda(lambda x: x.float()),  # in Float
    #transforms.Lambda(lambda x: x / torch.max(x)) #skallieren auf [0,1]
])



"""
Pre-Processing FUNKTIONS
"""


def add_gaus_noise(image, mean, sigma):
    noise = torch.randn_like(image) * sigma + mean
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image
 
 

"""
LOSS FUNKTIONS  +  EVALUATION FUNKTIONS
"""


#original Loss funktion from N2N paper
def loss_orig_n2n(noise_images, sigma, device, model):
    #src1 = add_gaus_noise(original, 0.5, sigma).to(device)
    #schöner 1 Zeiler:

    noise_image2 = (noise_images + torch.randn_like(noise_images) * (sigma+0.3)).to(device) #+ mean
    # Denoise image
    denoised = model(noise_images)
    loss_function = torch.nn.MSELoss()
    return loss_function(denoised, noise_image2), denoised, noise_image2


def loss_n2score(noise_images, sigma, device, model, methode):
    u = torch.randn_like(noise_images).to(device)
    sigma_a = noise_images
    noise = sigma_a*u
    noise_image = (original + noise).to(device)
    #loss, src1 = loss_SURE(src1, target, model, sigma)
    denoised = model(noise_image)
    if methode == "score_ar":
        loss = ((u + sigma_a*denoised)**2).sum().sqrt()
    else:
        loss = ((original - denoised)**2).sum().sqrt()
    return loss/original.numel(), denoised, noise_image


def loss_n2self(noise_image, batch_idx, model):
    masked_noise_image, mask = Mask.n2self_mask(noise_image, batch_idx)
    denoised = model(masked_noise_image)
    #j_invariant_denoised = n2self_infer_full_image(noise_image, model)  #TODO: weiß noch nicht was das ist und wofür es benutzt wird
    return torch.nn.MSELoss()(denoised*mask, noise_image*mask), denoised, masked_noise_image


def loss_n2void(noise_images, model, num_patches_per_img, windowsize, num_masked_pixels):
    patches = generate_patches_from_list(noise_images, num_patches_per_img=num_patches_per_img)
    mask  = Mask.n2void_mask(patches.shape, num_masked_pixels=8)


    masked_noise = Mask.excchange_in_mask_with_pixel_in_window(mask, patches, windowsize, num_masked_pixels)
    
    denoised = model(masked_noise)
    denoised_pixel = denoised * mask
    target_pixel = patches * mask
    
    loss_function = torch.nn.MSELoss() #TODO: richtigge Loss, funktion?
    return loss_function(denoised_pixel, target_pixel), denoised, patches


def n2n_loss_for_das(denoised, target):
    loss_function = torch.nn.MSELoss()
    loss = loss_function(target, denoised)
    #len(noisy) = anzahl an Bildern
    return 1/len(target) *loss #+c #c = varianze of noise

"""
TRAIN FUNKTIONS
"""



def train(model, optimizer, device, dataLoader, methode, sigma, mode, store, epoch, bestPsnr, writer):
    #mit lossfunktion aus "N2N(noiseToNoise)"
    #kabel ist gesplitet und ein Teil (src) wird im Model behandelt und der andere (target) soll verglichen werden
    loss_log = []
    psnr_log = []
    sim_log = []
    bestPsnr = bestPsnr
    bestSim = 0
    if mode=="test" or mode =="validate":
        model.eval()
    else:
        model.train()
    for batch_idx, (noise_images, label) in enumerate(tqdm(dataLoader)): #src.shape=[batchsize, rgb, w, h]
        noise_images = noise_images.to(device)
        if batch_idx == max_Iteration:
            break
        torch.set_grad_enabled(mode=="train")
    

        if methode == "n2n_orig":
            loss, denoised, noise_image2 = loss_orig_n2n(noise_images, sigma, device, model)
            
        elif "score" in methode:
            loss, denoised, noise_images = loss_n2score(noise_images, sigma, device, model, methode)
            #TODO: abhängig vom Noise, wird rekonstruiertes Endbild noch bearbeitet -> Suplemenrtary

        elif methode == "n2self":
            loss, denoised, noise_images = loss_n2self(noise_images, batch_idx, model)

        elif methode == "n2void":
            loss, denoised, noise_images = loss_orig_n2n(noise_images, model, num_patches_per_img=None, windowsize=5, num_masked_pixels=8)
            

       


        
        loss_log.append(loss.item())

        
        psnr_batch = Metric.calculate_psnr(noise_images, denoised)
        similarity_batch, diff_picture = Metric.calculate_similarity(noise_images, denoised)
        psnr_log.append(psnr_batch.item())
        sim_log.append(similarity_batch)
        if mode=="train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if psnr_batch.item() > bestPsnr:# or similarity_batch > bestSim:
                bestSim = similarity_batch
                bestPsnr = psnr_batch.item()
                print(f"best model found with psnr: {bestPsnr}")
                model_save_path = os.path.join(store, "models", f"{round(bestPsnr, 2)}NewBestRewardModel{epoch}-{batch_idx}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"saved new best model at path {model_save_path}")
                #denoised_norm = normalize_image(denoised)
                grid = make_grid(denoised, nrow=16, normalize=False) # Batch/number bilder im Raster
                writer.add_image('Denoised Images', grid, global_step=epoch * len(dataLoader) + batch_idx)
                show_tensor_as_picture(denoised)


    return loss_log, psnr_log, sim_log

        

def main(argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methode="n2n_orig"
    store_path = log_files()
    #run = wandb.init(entity="", project="my-project-name", anonymous="allow")
    writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))
    sigma=0.4

    celeba_dir = 'dataset/celeba_dataset'
    #dataset = datasets.CelebA(root=celeba_dir, split='train', download=True, transform=transform)
    #mean, std = calculate_mean_std_dataset(dataset, sigma=0.4)
    mean_training = torch.tensor([0.0047,0.0042,0.0041])
    std_training =  torch.tensor([0.0083, 0.0066, 0.0058])

    transform_noise = transforms.Compose([
        transforms.RandomResizedCrop((128,128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * sigma),  #Rauschen
        #transforms.Normalize(mean=mean_training, std=std_training) #Normaalisieren
        transforms.Lambda(lambda x: (x-x.min())  / (x.max() - x.min())),
        ])
    dataset = datasets.CelebA(root=celeba_dir, split='train', download=True, transform=transform_noise)
    dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)

    dataset_validate = datasets.CelebA(root=celeba_dir, split='valid', download=True, transform=transform_noise)
    dataLoader_validate = dataLoader = DataLoader(dataset_validate, batch_size=64, shuffle=True)

    mask = Mask.cut2self_mask((128,128), 64).to(device)

    model = N2N_Orig_Unet(3,3).to(device)
    #model = Cut2Self(mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    
    print(f"Using {device} device")
    #run.watch(model)
    bestPsnr = 0
    for epoch in range(max_Epochs):
        loss, psnr, similarity = train(model, optimizer, device, dataLoader, methode, sigma=sigma, mode="train", 
                                       store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer)

        for i, loss_item in enumerate(loss):
            writer.add_scalar('Train Loss', loss_item, epoch * len(dataLoader) + i)
            writer.add_scalar('Train PSNR', psnr[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Train Similarity', similarity[i], epoch * len(dataLoader) + i)
        bestPsnr = max(psnr)
        """
        if epoch%10 == 0:
            loss, psnr, similarity = train(model, optimizer, device, dataLoader_validate, methode, sigma=sigma, mode="validate")
            for i, loss_item in enumerate(loss):
                writer.add_scalar('Validation Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Validation PSNR', psnr[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Validation Similarity', similarity[i], epoch * len(dataLoader) + i)
        """

        

    #show_logs(loss, psnr, value_loss, value_psnr, similarity)



    print(loss)
    print(psnr)
    print(similarity)
    #show_pictures_from_dataset(dataset, model)
    writer.close()
    print("fertig\n")
    


if __name__ == "__main__":
    app.run(main)