import math
import os
import sys
import time
from pathlib import Path
#sys.path.insert(0, str(Path(__file__).resolve().parents[3])) #damit die Pfade auf dem Server richtig waren (copy past von PG)
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

import config as config

from models.N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self, U_Net_origi, U_Net, TestNet
from metric import Metric
from masks import Mask
from models.P_Unet import P_U_Net
from utils import *
from transformations import *


import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid


from absl import app
from torch.utils.tensorboard import SummaryWriter

max_Epochs = 20
modi = 0

def get_lr_lambda(initial_lr, step_size, lr_decrement):
    def lr_lambda(step):
        return max(0.0, initial_lr - (step // step_size) * lr_decrement / initial_lr)
    return lr_lambda
"""
lr = 0.001
lr_end = 0.00001
lambda_lr = lambda epoch: lr_end + (lr - lr_end) * (1 - epoch / max_Epochs)
"""


def saveModel_pictureComparison(model, len_dataloader, mode, store, epoch, bestPsnr, writer, save_model, batch_idx, original, batch, noise_images, denoised, psnr_batch):
    if round(psnr_batch.item(),1) > bestPsnr + 0.5 or batch_idx == len_dataloader-1:
        if round(psnr_batch.item(),1) > bestPsnr and mode != "test":
            bestPsnr = round(psnr_batch.item(),1)
            print(f"best model found with psnr: {bestPsnr}")
            model_save_path = os.path.join(store, "models", f"{round(bestPsnr, 1)}-{mode}-{epoch}-{batch_idx}.pth")
            if save_model:
                torch.save(model.state_dict(), model_save_path)
            else:
                f = open(model_save_path, "x")
                f.close()
        comparison = torch.cat((original[:4], denoised[:4], noise_images[:4]), dim=0)
        grid = make_grid(comparison, nrow=4, normalize=False).cpu()
        if mode == "train":
            writer.add_image('Denoised Images Training', grid, global_step=epoch * len_dataloader + batch_idx)
        elif mode == "validate":
            writer.add_image('Denoised Images Validation', grid, global_step=epoch * len_dataloader + batch_idx)
        else:
            writer.add_image('Denoised Images Test', grid, global_step=1 * len_dataloader + batch_idx)
    return bestPsnr

"""
TRAIN FUNKTIONS
"""
def get_loss(model, device, noise_images, noise_images2):
    denoised = model(noise_images)
    loss_function = torch.nn.MSELoss()
    return loss_function(denoised, noise_images2), denoised

def train(model, optimizer, device, dataLoader, sigma, mode, store, epoch, bestPsnr, writer, save_model):
    #mit lossfunktion aus "N2N(noiseToNoise)"
    #kabel ist gesplitet und ein Teil (src) wird im Model behandelt und der andere (target) soll verglichen werden
    global modi
    loss_log = []
    psnr_log = []
    original_psnr_log = []
    sim_log = []
    bestPsnr = bestPsnr
    for batch_idx, (original, label) in enumerate((dataLoader)):#tqdm
        original = original.to(device)
        batch = original.shape[0]
        noise_images, true_noise_sigma = add_noise_snr(original, snr_db=sigma)
        writer.add_scalar('True Noise sigma', true_noise_sigma, epoch * len(dataLoader) + batch_idx)
        noise_images = noise_images.to(device)
        noise_images2, true_noise_sigma = add_noise_snr(original, snr_db=sigma)
        if modi == 3:
            noise_images2, true_noise_sigma = add_noise_snr(noise_images, snr_db=sigma)

        if mode=="test" or mode =="validate":
            model.eval()
            with torch.no_grad():
                loss, denoised = get_loss(model, device, noise_images, noise_images2)             
        else:
            model.train()
            #original, noise_images are only important if n2void
            loss, denoised = get_loss(model, device, noise_images, noise_images2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #log Data
        original_psnr_batch = Metric.calculate_psnr(original, denoised)
        denoised = (denoised-denoised.min())  / (denoised.max() - denoised.min())
        noise_images = (noise_images-noise_images.min())  / (noise_images.max() - noise_images.min())
        original = (original-original.min())  / (original.max() - original.min())
        psnr_batch = Metric.calculate_psnr(original, denoised)
        similarity_batch, diff_picture = Metric.calculate_similarity(original, denoised)
        loss_log.append(loss.item())
        psnr_log.append(psnr_batch.item())
        original_psnr_log.append(original_psnr_batch.item())
        sim_log.append(similarity_batch)

        #save model + picture
        bestPsnr = saveModel_pictureComparison(model, len(dataLoader), mode, store, epoch, bestPsnr, writer, save_model, batch_idx, original, batch, noise_images, denoised, psnr_batch)

    return loss_log, psnr_log, sim_log, bestPsnr, original_psnr_log


def main(argv):
    print("Starte Programm!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"

    layout = {
        "Training vs Validation": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
            "PSNR": ["Multiline", ["PSNR/train", "PSNR/validation"]],
            "Similarity": ["Multiline", ["sim/train", "sim/validation"]],
        },
    }
    #writer = None
    sigma = config.sigma if config.useSigma else config.sigmadb
    save_model = False

    celeba_dir = config.celeba_dir
    
    transform_noise = transforms.Compose([
        #transforms.RandomResizedCrop((128,128)),
        transforms.CenterCrop((128,128)),
        #transforms.Resize((512,512)), #for self2self
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x:  x * 2 -1),
        ])
    print("lade Datensätze ...")
    dataset_all = datasets.CelebA(root=celeba_dir, split='train', download=False, transform=transform_noise)
    
    dataset_validate_all = datasets.CelebA(root=celeba_dir, split='valid', download=False, transform=transform_noise)

    dataset_test_all = datasets.CelebA(root=celeba_dir, split='test', download=False, transform=transform_noise)

    #create folder for all methods that will be run
    store_path_root = log_files()
    
    print(f"Using {device} device")
    
    #for methode in methoden_liste:
    global modi
    for i in range(3):

        #create to folders for loging details
        store_path = Path(os.path.join(store_path_root, str(i)))
        store_path.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "tensorboard"))
        tmp.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "models"))
        tmp.mkdir(parents=True, exist_ok=True)

        
        dataset = torch.utils.data.Subset(dataset_all, list(range(6400)))
        dataset_validate = torch.utils.data.Subset(dataset_validate_all, list(range(640)))
        dataset_test = torch.utils.data.Subset(dataset_test_all, list(range(640)))
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
        if torch.cuda.device_count() == 1:
            model = N2N_Orig_Unet(3,3).to(device) #default
            #model = P_U_Net(in_chanel=3, batchNorm=True, dropout=0.3).to(device)
            #model = U_Net().to(device)
        else:
            if modi == 0:
                model = U_Net(in_chanel=3, batchNorm=True).to(device)
            else:
                model = U_Net(in_chanel=3, batchNorm=False).to(device)
        #configAtr = getattr(config, methode) #config.methode wobei methode ein string ist
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003, betas = (0.9, 0.99), eps=10e-8)

        bestPsnr = -100
        bestPsnr_val = -100
        bestSim = -100
        bestSim_val = -100
        avg_train_psnr = []
        avg_val_psnr = []
        avg_test_psnr = []
        avg_train_sim = []
        avg_val_sim = []
        avg_test_sim = []
        #store_path = log_files()
        writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))
        writer.add_custom_scalars(layout)

        for epoch in tqdm(range(max_Epochs)):
            
            loss, psnr, similarity, bestPsnr, original_psnr_log = train(model, optimizer, device, dataLoader, sigma=sigma, mode="train", 
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer, save_model=save_model)
            
            for i, loss_item in enumerate(loss):
                writer.add_scalar('Train Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Train PSNR_iteration [0,1]', psnr[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Train Original PSNR (not normed)', original_psnr_log[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Train Similarity_iteration', similarity[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Train PSNR [0,1] avg', Metric.avg_list(psnr), epoch)
            
            if epoch % 5 == 0  or epoch==max_Epochs-1:
                model_save_path = os.path.join(store_path, "models", f"{epoch}-model.pth")
                if save_model:
                    torch.save(model.state_dict(), model_save_path)
                else:
                    f = open(model_save_path, "x")
                    f.close()

            if round(max(psnr),3) > bestPsnr:
                bestPsnr = round(max(psnr),3)
            if round(max(similarity),3) > bestSim:
                bestSim = round(max(similarity),3)
            
            print("Epochs highest PSNR: ", max(psnr))
            print("Epochs highest Sim: ", max(similarity))
            
                
            #runing on Server
            loss_val, psnr_val, similarity_val, bestPsnr_val, original_psnr_log_val = train(model, optimizer, device, dataLoader_validate, sigma=sigma, mode="validate", 
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer, save_model=save_model)

            for i, loss_item in enumerate(loss_val):
                writer.add_scalar('Validation Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Validation PSNR_iteration [0,1]', psnr_val[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Validation Original PSNR (not normed)', original_psnr_log_val[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Validation Similarity_iteration', similarity_val[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Validation PSNR [0,1] avg', Metric.avg_list(psnr_val), epoch)

            if max(psnr_val) > bestPsnr_val:
                bestPsnr_val = max(psnr_val)
            print("Loss: ", loss_val[-1])
            print("Epochs highest PSNR: ", max(psnr_val))
            print("Epochs highest Sim: ", max(similarity_val))
            avg_train_psnr.append(Metric.avg_list(psnr))
            avg_train_sim.append(Metric.avg_list(similarity))
            avg_val_psnr.append(Metric.avg_list(psnr_val))
            avg_val_sim.append(Metric.avg_list(similarity_val))
            writer.add_scalar("loss/train", Metric.avg_list(loss), epoch)
            writer.add_scalar("loss/validation", Metric.avg_list(loss_val), epoch)
            writer.add_scalar("PSNR/train", avg_train_psnr[-1], epoch)
            writer.add_scalar("PSNR/validation", avg_val_psnr[-1], epoch)
            writer.add_scalar("sim/train", avg_train_sim[-1], epoch)
            writer.add_scalar("sim/validation", avg_val_sim[-1], epoch)
        
            if math.isnan(loss[-1]):
                model_save_path = os.path.join(store_path, "models", "NaN.txt")
                f = open(model_save_path, "x")
                f.close()
                print(f"NAN, breche ab, break:")
                break

            if round(max(psnr_val),3) > bestPsnr_val:
                bestPsnr_val = round(max(psnr_val),3)
            if round(max(similarity_val),3) > bestSim_val:
                bestSim_val = round(max(similarity_val),3)
        
        #if torch.cuda.device_count() == 1:
            #continue
        loss_test, psnr_test, similarity_test, _, original_psnr_log_test = train(model, optimizer, device, dataLoader_test, sigma=sigma, mode="test", 
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer, save_model=save_model)
        writer.add_scalar("PSNR Test", Metric.avg_list(psnr_test), 0)
        writer.add_scalar("PSNR original (bot normed) Test", Metric.avg_list(original_psnr_log_test), 0)
        writer.add_scalar("Loss Test", Metric.avg_list(loss_test), 0)
        writer.add_scalar("Sim Test", Metric.avg_list(similarity_test), 0)

        modi +=1
    
    writer.close()
    print("fertig\n")
    print(str(datetime.now().replace(microsecond=0)).replace(':', '-'))
    

if __name__ == "__main__":
    app.run(main)