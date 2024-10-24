import math
import os
import sys
import time
from pathlib import Path
#sys.path.insert(0, str(Path(__file__).resolve().parents[3])) #damit die Pfade auf dem Server richtig waren (copy past von PG)
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

import config_test as config

from models.N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self, U_Net_origi, U_Net, TestNet
from metric import Metric
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
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\Code\DAS-denoiser-MA\runs"      #PC
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\runs"                           #vom Server
#tensorboard --logdir="C:\Users\LaAlexPi\Documents\01_Uni\MA\runs\"                         #Laptop



max_Iteration = 20
max_Epochs = 20
max_Predictions = 100 #für self2self um reconstruktion zu machen
#torch.manual_seed(42)

def get_lr_lambda(initial_lr, step_size, lr_decrement):
    def lr_lambda(step):
        return max(0.0, initial_lr - (step // step_size) * lr_decrement / initial_lr)
    return lr_lambda

def n2n_create_2_input(device, methode, original, noise_images):
    if "2_input" in methode:
        if config.useSigma:
            noise_image2 = add_norm_noise(original, config.sigma, -1,-1,False)
        else:
            noise_image2, alpha = add_noise_snr(original, snr_db=config.sigmadb)
    else:
        if config.useSigma:
            noise_image2 = add_norm_noise(noise_images, config.methodes['n2noise_1_input']['secoundSigma'], -1,-1,False)
        else:
            noise_image2, alpha = add_noise_snr(noise_images, snr_db=config.methodes['n2noise_1_input']['secoundSigma']) 
    noise_image2 = noise_image2.to(device)
    return noise_image2#original, noise_images  are onlly if n2void

def evaluateSigma(noise_image, vector):
    sigmas = torch.linspace(0.1, 0.7, 61)
    quality_metric = []
    for sigma in sigmas:
        
        simple_out = noise_image + sigma**2 * vector.detach()
        simple_out = (simple_out + 1) / 2
        quality_metric += [Metric.tv_norm(simple_out).item()]
    
    sigmas = sigmas.numpy()
    quality_metric = np.array(quality_metric)
    best_idx = np.argmin(quality_metric)
    return quality_metric[best_idx], sigmas[best_idx]


def saveModel_pictureComparison(model, len_dataloader, methode, mode, store, epoch, bestPsnr, writer, save_model, batch_idx, original, batch, noise_images, denoised, psnr_batch):
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
        if methode == "n2void" and mode == "train":
            skip = int(denoised.shape[0] / batch) #64=ursprüngllichhe Batchgröße
            denoised = denoised[::skip] # zeige nur jedes 6. Bild an (im path wird aus einem bild 6 wenn die Batchhgröße = 64)
            original = original[::skip]
            noise_images = noise_images[::skip]
        comparison = torch.cat((original[:4], denoised[:4], noise_images[:4]), dim=0)
        grid = make_grid(comparison, nrow=4, normalize=False).cpu()
        if mode == "train":
            writer.add_image('Denoised Images Training', grid, global_step=epoch * len_dataloader + batch_idx)
        elif mode == "validate":
            writer.add_image('Denoised Images Validation', grid, global_step=epoch * len_dataloader + batch_idx)
        else:
            writer.add_image('Denoised Images Test', grid, global_step=1 * len_dataloader + batch_idx)
    return bestPsnr


def exchange_in_mask_with_pixel_in_window(mask, data, windowsize, num_masked_pixels, replaceWithIself=False):
    """
    ersetzt die ausgewählten Pixel durch die Maske mit einem zufälligem Pixel in der Fenstergröße.
    Zentrum des Fensters ist das Pixel
    
    mask (tensor): Die benutzte Makse (batch, channels, height, width).
    data (tensor): Das benutzte Bild für die ersetzung der Pixel (batch, channels, height, width)
    windowsize (int): Quadratische Fenstergröße meistens 5x5
    num_masked_pixels (int): Die Anzahl der maskierten Pixel, die ausgewählt werden sollen.
    replaceWithIself (bool): is it possible to replalce tthe selectted Pixel with itself
    """
    cords = torch.nonzero(mask) #in jeder Zeile ist die Koordinate eines Wertes der nicht = 0 ist (also 1 z.b.)
    bearbeitete_Bilder = data.clone()
    memory = []
    for pixel_idx in range(cords.shape[0]): #cords.shape=(batch*num_mask_pixel*chanel, 4)   geht alle gefundenen Koordinaten durch die nicht 0 sind in Maske
        batch, chanel, x, y = cords[pixel_idx]
        batch, chanel, x, y = batch.item(), chanel.item(), x.item(), y.item()
        """
        OLD: when you want the same pixel masked in all chanels
        if chanel != 0: #copy the same pixel as in chanel 0
            new_x, new_y = memory[pixel_idx % num_masked_pixels]
            bearbeitete_Bilder[batch, chanel, x, y] = data[batch, chanel, new_x, new_y]
        else: 
        """
        while True: 
            #      max ( 0, min(width-1, x + rand(-window/2, window/2+1)) )
            new_x = max(0, min(bearbeitete_Bilder.shape[2] - 1, x + torch.randint(-windowsize//2, windowsize//2 + 1, (1,)).item()))
            new_y = max(0, min(bearbeitete_Bilder.shape[2] - 1, y + torch.randint(-windowsize//2, windowsize//2 + 1, (1,)).item()))
            #don't replace with the same pixel
            if (new_x, new_y) != (x,y) or replaceWithIself:
                break
        memory.append((new_x, new_y))
        bearbeitete_Bilder[batch, chanel, x, y] = data[batch, chanel, new_x, new_y]
    return bearbeitete_Bilder
def select_random_pixels_new(image_shape, num_masked_pixels):
    num_pixels = image_shape[0] * image_shape[1] * image_shape[2]
    # Erzeuge zufällige Indizes für die ausgewählten maskierten Pixel
    masked_indices = torch.randperm(num_pixels)[:num_masked_pixels]
    mask = torch.zeros(image_shape[0], image_shape[1], image_shape[2])
    # Pixel in Maske auf 1 setzen
    mask.view(-1)[masked_indices] = 1
    # Mache für alle Chanels
    #mask = mask.unsqueeze(0)
    return mask
def select_random_pixels_old(image_shape, num_masked_pixels): #presi3
    num_pixels = image_shape[1] * image_shape[2]
    # Erzeuge zufällige Indizes für die ausgewählten maskierten Pixel
    masked_indices = torch.randperm(num_pixels)[:num_masked_pixels]
    mask = torch.zeros(image_shape[1], image_shape[2])
    # Pixel in Maske auf 1 setzen
    mask.view(-1)[masked_indices] = 1
    # Mache für alle Chanels
    mask = mask.unsqueeze(0).expand(image_shape[0], -1, -1)
    return mask

def n2void(original_images, noise_images, model, device, num_patches_per_img, windowsize, num_masked_pixels, augmentation, methode):
    patches, clean_patches = generate_patches_from_list(noise_images, original_images, num_patches_per_img=num_patches_per_img, augment=augmentation)
    if 'new' in methode:
        if 'scaled' in methode:
            if len(patches.shape)==3:
                mask = select_random_pixels_new(patches.shape, num_masked_pixels*3)
            else:
                mask_for_batch = []
                for i in range(patches.shape[0]):
                    mask_for_batch.append(select_random_pixels_new((patches.shape[1],patches.shape[2],patches.shape[3]), num_masked_pixels*3))
                mask = torch.stack(mask_for_batch)
        else:
            if len(patches.shape)==3:
                mask = select_random_pixels_new(patches.shape, num_masked_pixels)
            else:
                mask_for_batch = []
                for i in range(patches.shape[0]):
                    mask_for_batch.append(select_random_pixels_new((patches.shape[1],patches.shape[2],patches.shape[3]), num_masked_pixels))
                mask = torch.stack(mask_for_batch)
    else: #presi 3
        if len(patches.shape)==3:
            mask = select_random_pixels_old(patches.shape, num_masked_pixels)
        else:
            mask_for_batch = []
            for i in range(patches.shape[0]):
                mask_for_batch.append(select_random_pixels_old((patches.shape[1],patches.shape[2],patches.shape[3]), num_masked_pixels))
            mask = torch.stack(mask_for_batch)
    #mask, num_masked_pixels_total = Mask.mask_random(patches, 8, mask_size=(1,1))
    mask = mask.to(device)
    masked_noise = exchange_in_mask_with_pixel_in_window(mask, patches, windowsize, num_masked_pixels) #TODO
    
    denoised = model(masked_noise)
    denoised_pixel = denoised * mask
    target_pixel = patches * mask
    
    loss_function = torch.nn.MSELoss()
    loss = torch.mean((denoised_pixel - target_pixel)**2)
    return loss_function(denoised_pixel, target_pixel), denoised, patches, clean_patches

def calculate_loss(model, device, methode, original, noise_images, augmentation):
    lr = 0
    ud = 0
    est_sigma_opt = -1
    #normalise Data as in github
    mean_noise = noise_images.mean(dim=[0,2,3])
    std_noise = noise_images.std(dim=[0,2,3])
    noise_images = (noise_images - mean_noise[None, :, None, None]) / std_noise[None, :, None, None]
    #mean_clean = original.mean(dim=[0,2,3])
    #std_clean = original.std(dim=[0,2,3])
    #original = (original - mean_clean[None, :, None, None]) / std_clean[None, :, None, None]

    loss, denoised, patches, original_patches = n2void(original, noise_images, model, device, num_patches_per_img=None, windowsize=5, num_masked_pixels=8, augmentation=True, methode=methode)
    noise_images = patches
    original = original_patches

    return loss, denoised, original, noise_images, (lr, ud, est_sigma_opt)


"""
TRAIN FUNKTIONS
"""

def train(model, optimizer, scheduler, device, dataLoader, methode, sigma, mode, store, epoch, bestPsnr, writer, save_model, sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True):
    #mit lossfunktion aus "N2N(noiseToNoise)"
    #kabel ist gesplitet und ein Teil (src) wird im Model behandelt und der andere (target) soll verglichen werden
    loss_log = []
    psnr_log = []
    original_psnr_log = []
    sim_log = []
    best_sigmas = []    #only for Score in validation + test
    all_tvs = []     #only for Score in validation + test
    true_sigma_score = []
    bestPsnr = bestPsnr
    bestSim = -1
    n = torch.tensor([]).reshape(0, 3*128*128).to(device) #n like in n2info for estimate sigma
    lex = 0
    lin = 0
    all_marked = 0
    for batch_idx, (original, label) in enumerate((dataLoader)):#tqdm
        original = original.to(device)
        batch = original.shape[0]
        if config.useSigma:
            noise_images = add_norm_noise(original, sigma, a=-1, b=1, norm=False)
            true_noise_sigma = sigma
        else:
            noise_images, true_noise_sigma = add_noise_snr(original, snr_db=sigma)
        writer.add_scalar('True Noise sigma', true_noise_sigma, epoch * len(dataLoader) + batch_idx)
        noise_images = noise_images.to(device)

        if mode=="test" or mode =="validate":
            model.eval()
            with torch.no_grad():
                loss, _, _, _, optional_tuples = calculate_loss(model, device, methode, original, noise_images, augmentation)
                (_, _, est_sigma_opt) = optional_tuples
                
                #calculate mean and std for each Image in batch in every chanal
                #mean = noise_images.mean(dim=[0,2,3])
                #std = noise_images.std(dim=[0,2,3])
                #noise_images = (noise_images - mean[None, :, None, None]) / std[None, :, None, None]
                noise_images = (noise_images - noise_images.mean(dim=(1, 2), keepdim=True)) / noise_images.std(dim=(1, 2), keepdim=True)
                denoised = model(noise_images)
                
                        
        else:
            model.train()
            #original, noise_images are only important if n2void
            loss, denoised, original, noise_images, optional_tuples = calculate_loss(model, device, methode, original, noise_images, augmentation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler was for use in n2same but i have problems with this implementation
            #if config.methodes[methode]['sheduler']:
                #scheduler.step()

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
        bestPsnr = saveModel_pictureComparison(model, len(dataLoader), methode, mode, store, epoch, bestPsnr, writer, save_model, batch_idx, original, batch, noise_images, denoised, psnr_batch)
    
    
    return loss_log, psnr_log, sim_log, bestPsnr, sigma_info, original_psnr_log



        

def main(argv):
    print("Starte Programm!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    methoden_liste = ["n2void old", "n2void new", "n2void new scaled"]

    layout = {
        "Training vs Validation": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
            "PSNR": ["Multiline", ["PSNR/train", "PSNR/validation"]],
            "Similarity": ["Multiline", ["sim/train", "sim/validation"]],
        },
    }
    #writer = None
    sigma = config.sigma if config.useSigma else config.sigmadb
    save_model = config.save_model #save models as pth

    celeba_dir = config.celeba_dir

    end_results = pd.DataFrame(columns=methoden_liste)
    #end_results = pd.DataFrame(columns=config.methodes.keys())

    
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

    
    for methode in methoden_liste:
    #for methode, method_params in config.methodes.items():

        #create to folders for loging details
        store_path = Path(os.path.join(store_path_root, methode))
        store_path.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "tensorboard"))
        tmp.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "models"))
        tmp.mkdir(parents=True, exist_ok=True)

        print(methode)

        dataset = torch.utils.data.Subset(dataset_all, list(range(1056)))
        dataset_validate = torch.utils.data.Subset(dataset_validate_all, list(range(128)))
        dataset_test = torch.utils.data.Subset(dataset_test_all, list(range(128)))
        
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
        if torch.cuda.device_count() == 1:
            model = N2N_Orig_Unet(3,3).to(device) #default
            #model = P_U_Net(in_chanel=3, batchNorm=True, dropout=0.3).to(device)
            #model = U_Net().to(device)
        else:
            model = U_Net(in_chanel=3, batchNorm=True).to(device)
        #configAtr = getattr(config, methode) #config.methode wobei methode ein string ist
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
        scheduler = None
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
        sigma_info = 1 #noise2Info starting pointt for estimated sigma

        for epoch in tqdm(range(max_Epochs)):
            
            loss, psnr, similarity, bestPsnr, sigma_info, original_psnr_log = train(model, optimizer, scheduler, device, dataLoader, methode, sigma=sigma, mode="train", 
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer, save_model=save_model, 
                                        sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True)
            
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
            
            #if torch.cuda.device_count() == 1:
                #continue
            
            #if epoch%100!=0:
                #continue
                
            #runing on Server
            loss_val, psnr_val, similarity_val, bestPsnr_val, sigma_info, original_psnr_log_val = train(model, optimizer, scheduler, device, dataLoader_validate, methode, sigma=sigma, mode="validate", 
                                                                    store=store_path, epoch=epoch, bestPsnr=bestPsnr_val, writer = writer, 
                                                                    save_model=save_model, sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True)

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
                print(f"NAN, breche ab, break: {methode}")
                break

            if round(max(psnr_val),3) > bestPsnr_val:
                bestPsnr_val = round(max(psnr_val),3)
            if round(max(similarity_val),3) > bestSim_val:
                bestSim_val = round(max(similarity_val),3)
        
        #if torch.cuda.device_count() == 1:
            #continue
        loss_test, psnr_test, similarity_test, _, _, original_psnr_log_test = train(model, optimizer, scheduler, device, dataLoader_test, methode, sigma=sigma, mode="test", 
                                                                    store=store_path, epoch=-1, bestPsnr=-1, writer = writer, 
                                                                    save_model=False, sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True)
        writer.add_scalar("PSNR Test", Metric.avg_list(psnr_test), 0)
        writer.add_scalar("PSNR original (bot normed) Test", Metric.avg_list(original_psnr_log_test), 0)
        writer.add_scalar("Loss Test", Metric.avg_list(loss_test), 0)
        writer.add_scalar("Sim Test", Metric.avg_list(similarity_test), 0)

        model_save_path = os.path.join(store_path, "models", f"last-model-{methode}.pth")
        torch.save(model.state_dict(), model_save_path)

        end_results[methode] = [loss[-1], 
                                bestPsnr, 
                                avg_train_psnr[-1], 
                                round(bestSim,3), 
                                round(avg_train_sim[-1],3),
                                bestPsnr_val, 
                                avg_val_psnr[-1], 
                                round(bestSim_val,3), 
                                round(avg_val_sim[-1],3),
                                round(max(psnr_test),3), 
                                round(max(similarity_test),3)]

    end_results.index = ['Loss', 
                         'Max PSNR Training', 
                         'Avg PSNR last Training', 
                         'SIM Training', 
                         'Avg SIM last Training',
                         'PSNR Validation', 
                         'Avg PSNR last Training', 
                         'SIM Validation', 
                         'Avg SIM last Validation',
                         'PSNR Test', 
                         'SIM Test']

    #show_logs(loss, psnr, value_loss, value_psnr, similarity)
    end_results.to_csv(os.path.join(store_path_root, "result_tabel.csv"))
    print(end_results)
    print(sigma_info)
    #show_pictures_from_dataset(dataset, model)
    writer.close()
    print("fertig\n")
    print(str(datetime.now().replace(microsecond=0)).replace(':', '-'))
    

if __name__ == "__main__":
    app.run(main)