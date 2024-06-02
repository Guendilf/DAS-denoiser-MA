import math
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

from models.N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self, U_Net_origi, U_Net, TestNet
from metric import Metric
from masks import Mask
from utils import show_logs, show_pictures_from_dataset, log_files, show_tensor_as_picture,  normalize_image, add_norm_noise
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
max_Epochs = 20



"""
Pre-Processing FUNKTIONS
"""
 

"""
LOSS FUNKTIONS  +  EVALUATION FUNKTIONS
"""


def loss_n2score(noise_images, sigma_min, sigma_max, q, device, model, methode, sigma): #q=batchindex/dataset
    u = torch.randn_like(noise_images).to(device)
    sigma_a = sigma_max*(1-q) + sigma_min*q
    noise = sigma_a*u
    #loss, src1 = loss_SURE(src1, target, model, sigma)
    denoised = model(noise_images+noise)
    if methode == "score_ar":
        loss = ((u + sigma_a*denoised)**2).mean()
    else:
        mse = torch.nn.MSELoss()
        output_f = ((sigma/255)**2)*denoised
        recon = output_f + noise_images+noise
        loss = mse(denoised *sigma, -u)
        loos2 = mse(recon, noise_images)
    return loss, denoised, noise_images



"""
TRAIN FUNKTIONS
"""



def train(model, optimizer, device, dataLoader, methode, sigma, mode, store, epoch, bestPsnr, writer, min_value, max_value, min_value2, max_value2):
    #mit lossfunktion aus "N2N(noiseToNoise)"
    #kabel ist gesplitet und ein Teil (src) wird im Model behandelt und der andere (target) soll verglichen werden
    loss_log = []
    psnr_log = []
    sim_log = []
    psnr_orig_log = []
    bestPsnr = bestPsnr
    bestSim = -1
    if mode=="test" or mode =="validate":
        model.eval()
    else:
        model.train()
    for batch_idx, (original, label) in enumerate(tqdm(dataLoader)): #src.shape=[batchsize, rgb, w, h]
        original = original.to(device)
        if "norm" in methode:
            noise_images = add_norm_noise(original, sigma, min_value, max_value, a=-1, b=1)
        else:
            noise_images = add_norm_noise(original, sigma, min_value, max_value, a=-1, b=1, norm=False)
        noise_images = noise_images.to(device)
        #if batch_idx == max_Iteration:
            #break
        torch.set_grad_enabled(mode=="train")

        loss, denoised, noise_images = loss_n2score(noise_images, sigma_min=1, sigma_max=30, q=epoch/max_Epochs, #q=batch_idx/len(dataLoader), 
                                                    device=device, model=model, methode=methode, sigma=sigma)
        if "tweedie" in methode:
            denoised = noise_images + sigma**2*denoised

            
        
        psnr_original = Metric.calculate_psnr(original, denoised)
        denoised = (denoised-denoised.min())  / (denoised.max() - denoised.min())
        noise_images = (noise_images-noise_images.min())  / (noise_images.max() - noise_images.min())
        original = (original-original.min())  / (original.max() - original.min())


        
        loss_log.append(loss.item())

        
        psnr_batch = Metric.calculate_psnr(original, denoised)
        similarity_batch, diff_picture = Metric.calculate_similarity(original, denoised)
        psnr_orig_log.append(psnr_original.item())
        psnr_log.append(psnr_batch.item())
        sim_log.append(similarity_batch)
        if mode=="train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if round(psnr_batch.item(),1) > bestPsnr or batch_idx == len(dataLoader)-1:# or similarity_batch > bestSim:
                if round(psnr_batch.item(),1) > bestPsnr:
                    bestSim = similarity_batch
                    bestPsnr = round(psnr_batch.item(),1)
                    print(f"best model found with psnr: {bestPsnr}")
                    model_save_path = os.path.join(store, "models", f"{round(bestPsnr, 1)}NewBestRewardModel{epoch}-{batch_idx}.pth")
                    print(f"saved new best model at path {model_save_path}")
                    #denoised_norm = normalize_image(denoised)
                
                grid = make_grid(denoised, nrow=16, normalize=False) # Batch/number bilder im Raster
                writer.add_image('Denoised Images', grid, global_step=epoch * len(dataLoader) + batch_idx)
                #show_tensor_as_picture(denoised)
            
    print("epochs last psnr: ", psnr_log[-1])
    print("epochs last sim: ", sim_log[-1])
    
    return loss_log, psnr_log, sim_log, bestPsnr, psnr_orig_log

        

def main(argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cuda:3"

    methoden_liste = ["score_ar tweedie norm", "score tweedie norm", "score_ar tweedie", "score tweedie", "score_ar tweedie range", "score tweedie range"]

    
    layout = {
        "Training vs Validation": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
            "PSNR": ["Multiline", ["PSNR/train", "PSNR/validation"]],
            "Similarity": ["Multiline", ["sim/train", "sim/validation"]],
        },
    }
    
    #writer = None
    sigma=0.4

    celeba_dir = 'dataset/celeba_dataset'    

    transform_noise = transforms.Compose([
        transforms.RandomResizedCrop((128,128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x:  x * 2 -1), #[-1,1]
        ])
    transform_noise2 = transforms.Compose([
        transforms.RandomResizedCrop((128,128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        ])
    print("lade Datensätze")
    dataset = datasets.CelebA(root=celeba_dir, split='train', download=False, transform=transform_noise)
    dataset = torch.utils.data.Subset(dataset, list(range(6400)))
    

    dataset_validate = datasets.CelebA(root=celeba_dir, split='valid', download=False, transform=transform_noise)
    dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(640)))


    
    print(f"Using {device} device")
    #run.watch(model)
    
    for methode in methoden_liste:

        if "range" in methode:
            dataset = datasets.CelebA(root=celeba_dir, split='train', download=False, transform=transform_noise2)
            dataset = torch.utils.data.Subset(dataset, list(range(6400)))
            dataset_validate = datasets.CelebA(root=celeba_dir, split='valid', download=False, transform=transform_noise2)
            dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(640)))
        
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=True)
        model = N2N_Orig_Unet(3,3).to(device)
        #model = TestNet(3,3).to(device)
        #model = U_Net().to(device)
        #model = Cut2Self(mask).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        bestPsnr = -100
        bestPsnr_val = -100
        store_path = log_files()
        writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))
        writer.add_custom_scalars(layout)

        for epoch in tqdm(range(max_Epochs)):
            loss, psnr, similarity, bestPsnr, psnr_orig = train(model, optimizer, device, dataLoader, methode, sigma=sigma, mode="train", 
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer, min_value=-3.25, max_value=3.25, min_value2=-5.67, max_value2=5.83)

            
            for i, loss_item in enumerate(loss):
                writer.add_scalar('Train Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Train PSNR_iteration [0,1]', psnr[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Train PSNR_iteration [-1,1]', psnr_orig[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Train Similarity_iteration', similarity[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Train PSNR [0,1] avg', Metric.avg_list(psnr), epoch)
            writer.add_scalar('Train PSNR [-1,1] avg', Metric.avg_list(psnr_orig), epoch)
            
            high_psnr = max(psnr)
            high_sim = max(similarity)
            if epoch % 5 == 0 or round(max(psnr),1) > bestPsnr:
                if round(max(psnr),1) > bestPsnr:
                    bestPsnr = round(max(psnr),1)
                model_save_path = os.path.join(store_path, "models", f"{epoch}-model.pth")
            print("Epochs highest PSNR: ", high_psnr)
            print("Epochs highest Sim: ", high_sim)
            
            
            #if epoch%10 == 0 or epoch == max_Epochs:
            loss_val, psnr_val, similarity_val, bestPsnr_val, psnr_orig_val = train(model, optimizer, device, dataLoader_validate, methode, sigma=sigma, mode="validate",
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr_val, writer = writer, min_value=-3.2, max_value=3.15, min_value2=-5.4, max_value2=5.32)
            high_psnr = -100
            high_sim = -100
            for i, loss_item in enumerate(loss_val):
                writer.add_scalar('Validation Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Validation PSNR_iteration [0,1]', psnr_val[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Validation PSNR_iteration [-1,1]', psnr_orig_val[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Validation Similarity_iteration', similarity_val[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Validation PSNR [0,1] avg', Metric.avg_list(psnr_val), epoch)
            writer.add_scalar('Validation PSNR [-1,1] avg', Metric.avg_list(psnr_orig_val), epoch)
            high_psnr = max(psnr_val)
            high_sim = max(similarity_val)
            if max(psnr_val) > bestPsnr_val:
                bestPsnr_val = max(psnr_val)
            print("Epochs highest PSNR: ", high_psnr)
            print("Epochs highest Sim: ", high_sim)
            writer.add_scalar("loss/train", Metric.avg_list(loss), epoch)
            writer.add_scalar("loss/validation", Metric.avg_list(loss_val), epoch)
            writer.add_scalar("PSNR/train", Metric.avg_list(psnr), epoch)
            writer.add_scalar("PSNR/validation", Metric.avg_list(psnr_val), epoch)
            writer.add_scalar("sim/validation", Metric.avg_list(similarity), epoch)
            writer.add_scalar("sim/validation", Metric.avg_list(similarity_val), epoch)

            print("Best PSNR: ", bestPsnr)
            print("Best Val_PSNR: ", bestPsnr_val)
            print()
            

    #show_pictures_from_dataset(dataset, model)
    writer.close()
    print("fertig\n")
    


if __name__ == "__main__":
    app.run(main)