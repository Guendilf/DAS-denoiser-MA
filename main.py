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
from utils import *
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
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\Code\DAS-denoiser-MA\runs"      #PC
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\runs"                           #vom Server



max_Iteration = 2
max_Epochs = 1



"""
Pre-Processing FUNKTIONS
"""



 
 

"""
LOSS FUNKTIONS  +  EVALUATION FUNKTIONS
"""


#original Loss funktion from N2N paper
def loss_n2noise(original, noise_images, sigma, device, model, min_value, max_value, a=-1, b=1):
    #src1 = add_gaus_noise(original, 0.5, sigma).to(device)
    #schöner 1 Zeiler:
    noise_image2 = add_norm_noise(original, sigma+0.3, min_value, max_value, a=-1, b=1)
    noise_image2 = noise_image2.to(device) #+ mean
    # Denoise image
    denoised = model(noise_images)
    loss_function = torch.nn.MSELoss()
    return loss_function(denoised, noise_image2), denoised, noise_image2


def loss_n2score(noise_images, sigma_min, sigma_max, q, device, model, methode): #q=batchindex/dataset
    u = torch.randn_like(noise_images).to(device)
    sigma_a = sigma_max*(1-q) + sigma_min*q
    vectorMap = model(noise_images+sigma_a*u)
    loss = torch.mean((sigma_a * vectorMap + u)**2)
    return loss, vectorMap


def loss_n2self(noise_image, batch_idx, model):
    masked_noise_image, mask = Mask.n2self_mask(noise_image, batch_idx)
    denoised = model(masked_noise_image)
    #j_invariant_denoised = n2self_infer_full_image(noise_image, model)  #TODO: weiß noch nicht was das ist und wofür es benutzt wird
    return torch.nn.MSELoss()(denoised*mask, noise_image*mask), denoised, masked_noise_image


def loss_n2void(original_images, noise_images, model, device, num_patches_per_img, windowsize, num_masked_pixels):
    patches, clean_patches = generate_patches_from_list(noise_images, original_images, num_patches_per_img=num_patches_per_img)
    mask  = Mask.n2void_mask(patches.shape, num_masked_pixels=8).to(device)
    masked_noise = Mask.exchange_in_mask_with_pixel_in_window(mask, patches, windowsize, num_masked_pixels)
    
    denoised = model(masked_noise)
    denoised_pixel = denoised * mask
    target_pixel = patches * mask
    
    loss_function = torch.nn.MSELoss() #TODO: richtigge Loss, funktion?
    return loss_function(denoised_pixel, target_pixel), denoised, patches, clean_patches

def loss_n2same(noise_images, device, model, lambda_inv=2):
    mask, marked_points = (Mask.cut2self_mask((noise_images.shape[2],noise_images.shape[3]), noise_images.shape[0], mask_size=(1, 1), mask_percentage=0.0005)) #0,5% Piel maskieren
    mask = mask.to(device)
    mask = mask.unsqueeze(1)  # (b, 1, w, h)
    mask = mask.expand(-1, 3, -1, -1) # (b, 3, w, h)
    masked_input = (1-mask) * noise_images #delete masked pixels in noise_img
    denoised = model(noise_images)
    denoised_mask = model(masked_input)
    mse = torch.nn.MSELoss()
    loss_rec = torch.mean((denoised-noise_images)**2) # mse(denoised, noise_images)
    loss_inv = torch.sum(mask*(denoised-denoised_mask)**2)# mse(denoised, denoised_mask)
    loss = loss_rec + lambda_inv * (loss_inv/marked_points).sqrt()
    return loss, denoised, denoised_mask #J = count of maked_points

def n2n_loss_for_das(denoised, target):
    loss_function = torch.nn.MSELoss()
    loss = loss_function(target, denoised)
    #len(noisy) = anzahl an Bildern
    return 1/len(target) *loss #+c #c = varianze of noise

def calculate_loss(model, device, dataLoader, methode, sigma, min_value2, max_value2, batch_idx, original, noise_images):
    if methode == "n2noise":
        loss, denoised, noise_image2 = loss_n2noise(original, noise_images, sigma, device, model, min_value2, max_value2)
    elif methode == "n2score":
        loss, denoised = loss_n2score(noise_images, sigma_min=1, sigma_max=30, q=batch_idx/len(dataLoader), 
                                                        device=device, model=model, methode=methode)
        #tweedie - funktion for Gaus
        denoised = noise_images + sigma**2*denoised

    elif methode == "n2self":
        loss, denoised, masked_noise_image = loss_n2self(noise_images, batch_idx, model)

    elif methode == "n2void":
        #normalise Data as in github
        mean_noise = noise_images.mean(dim=[0,2,3])
        std_noise = noise_images.std(dim=[0,2,3])
        noise_images = (noise_images - mean_noise[None, :, None, None]) / std_noise[None, :, None, None]
        mean_clean = original.mean(dim=[0,2,3])
        std_clean = original.std(dim=[0,2,3])
        original = (original - mean_clean[None, :, None, None]) / std_clean[None, :, None, None]

        loss, denoised, patches, original_patches = loss_n2void(original, noise_images, model, device, num_patches_per_img=None, windowsize=5, num_masked_pixels=8)
        noise_images = patches
        original = original_patches
    elif methode == "n2same":
        #normalise Data as in github
        mean_noise = noise_images.mean(dim=[0,2,3])
        std_noise = noise_images.std(dim=[0,2,3])
        noise_images = (noise_images - mean_noise[None, :, None, None]) / std_noise[None, :, None, None]

        loss, denoised, denoised_mask = loss_n2same(noise_images, device, model, lambda_inv=2)
    return loss, denoised, original, noise_images #original, noise_images  are onlly if n2void

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
    for batch_idx, (original, label) in enumerate(tqdm(dataLoader)):
        original = original.to(device)
        #noise_images = add_norm_noise(original, sigma, min_value, max_value, a=-1, b=1)
        noise_images = add_noise_snr(original, snr_db=2)
        noise_images = noise_images.to(device)
        #get specific values for training and validation
        if mode=="test" or mode =="validate":
            model.eval()
            with torch.no_grad():
                loss, _, _, _ = calculate_loss(model, device, dataLoader, methode, sigma, min_value2, max_value2, batch_idx, original, noise_images)
                if methode == "n2noise":
                    denoised = model (noise_images)            
                elif methode == "n2score":
                    denoised = noise_images + sigma**2 * model(noise_images)
                elif methode == "n2self":
                    if "j-invariant" in methode:
                        denoised = Mask.n2self_jinv_recon(noise_images, model)
                    else:
                        denoised = model(noise_images)
                elif methode == "n2void" or methode == "n2same":
                    #calculate mean and std for each Image in batch in every chanal
                    mean = noise_images.mean(dim=[0,2,3])
                    std = noise_images.std(dim=[0,2,3])
                    noise_images = (noise_images - mean[None, :, None, None]) / std[None, :, None, None]
                    denoised = model(noise_images)
                    #normalization backwords
                    denoised = (denoised * std[None, :, None, None]) / mean[None, :, None, None]

        else:
            model.train()
            #original, noise_images are only important if n2void
            loss, denoised, original, noise_images = calculate_loss(model, device, dataLoader, methode, sigma, min_value2, max_value2, batch_idx, original, noise_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #log Data
        psnr_original = Metric.calculate_psnr(original, denoised)
        denoised = (denoised-denoised.min())  / (denoised.max() - denoised.min())
        noise_images = (noise_images-noise_images.min())  / (noise_images.max() - noise_images.min()) #wird nicht benutzt
        original = (original-original.min())  / (original.max() - original.min())

        loss_log.append(loss.item())

        psnr_batch = Metric.calculate_psnr(original, denoised)
        similarity_batch, diff_picture = Metric.calculate_similarity(original, denoised)
        psnr_orig_log.append(psnr_original.item())
        psnr_log.append(psnr_batch.item())
        sim_log.append(similarity_batch)
        if round(psnr_batch.item(),1) > bestPsnr or batch_idx == len(dataLoader)-1:# or similarity_batch > bestSim:
            if round(psnr_batch.item(),1) > bestPsnr:
                bestSim = similarity_batch
                bestPsnr = round(psnr_batch.item(),1)
                print(f"best model found with psnr: {bestPsnr}")
                model_save_path = os.path.join(store, "models", f"{round(bestPsnr, 1)}-{mode}-{epoch}-{batch_idx}.pth")
                torch.save(model.state_dict(), model_save_path)
                #print(f"saved new best model at path {model_save_path}")
            if methode == "n2void" and mode == "train":
                batch = original.shape[0]
                skip = int(denoised.shape[0] / batch) #64=ursprüngllichhe Batchgröße
                denoised = denoised[::skip] # zeige nur jedes 6. Bild an (im path wird aus einem bild 6 wenn die Batchhgröße = 64)
            grid = make_grid(denoised, nrow=16, normalize=False) # Batch/number bilder im Raster
            if mode == "train":
                writer.add_image('Denoised Images Training', grid, global_step=epoch * len(dataLoader) + batch_idx)
            else:
                writer.add_image('Denoised Images Validation', grid, global_step=epoch * len(dataLoader) + batch_idx)
            #show_tensor_as_picture(denoised)

    print("epochs last psnr: ", psnr_log[-1])
    print("epochs last sim: ", sim_log[-1])
    
    return loss_log, psnr_log, sim_log, bestPsnr, psnr_orig_log

        

def main(argv):
    print("Starte Programm!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    methoden_liste = ["n2noise", "n2score", "n2self", "n2same", "n2void"]
    #methode = methoden_liste[5]
    
    #run = wandb.init(entity="", project="my-project-name", anonymous="allow")
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

    mean_training = torch.tensor([ 0.0010, -0.0023, -0.0040])
    std_training =  torch.tensor([0.0112, 0.0104, 0.0103])
    
    transform_noise = transforms.Compose([
        transforms.RandomResizedCrop((128,128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x:  x * 2 -1),
        ])
    print("lade Datensätze ...")
    dataset = datasets.CelebA(root=celeba_dir, split='train', download=False, transform=transform_noise)
    dataset = torch.utils.data.Subset(dataset, list(range(6400)))
    
    dataset_validate = datasets.CelebA(root=celeba_dir, split='valid', download=False, transform=transform_noise)
    dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(640)))
    
    print(f"Using {device} device")
    #run.watch(model)
    
    for methode in methoden_liste:
        if methode == "n2void":
            dataset = torch.utils.data.Subset(dataset, list(range(1056)))
            dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(128)))
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=True)
        if torch.cuda.device_count() == 1:
            model = N2N_Orig_Unet(3,3).to(device)
        else:
            #model = TestNet(3,3).to(device)
            model = U_Net().to(device)
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
                torch.save(model.state_dict(), model_save_path)
            print("Epochs highest PSNR: ", high_psnr)
            print("Epochs highest Sim: ", high_sim)
            
            if torch.cuda.device_count() == 1:
                continue
            
            #runing on Server
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

    #show_logs(loss, psnr, value_loss, value_psnr, similarity)

    print(loss)
    print(psnr)
    print(similarity)
    #show_pictures_from_dataset(dataset, model)
    writer.close()
    print("fertig\n")
    

if __name__ == "__main__":
    app.run(main)