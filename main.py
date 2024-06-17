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
from loss import calculate_loss

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
max_Epochs = 20

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
"""
TRAIN FUNKTIONS
"""

def train(model, optimizer, device, dataLoader, methode, sigma, mode, store, epoch, bestPsnr, writer, min_value, max_value, min_value2, max_value2, save_model):
    #mit lossfunktion aus "N2N(noiseToNoise)"
    #kabel ist gesplitet und ein Teil (src) wird im Model behandelt und der andere (target) soll verglichen werden
    loss_log = []
    psnr_log = []
    sim_log = []
    psnr_orig_log = []
    best_sigmas = []    #only for Score in validation + test
    all_tvs = []     #only for Score in validation + test
    bestPsnr = bestPsnr
    bestSim = -1
    for batch_idx, (original, label) in enumerate(tqdm(dataLoader)):
        original = original.to(device)
        batch = original.shape[0]
        #noise_images = add_norm_noise(original, sigma, min_value, max_value, a=-1, b=1)
        noise_images, alpha = add_noise_snr(original, snr_db=sigma)
        sigma = alpha
        noise_images = noise_images.to(device)
        #get specific values for training and validation
        if mode=="test" or mode =="validate":
            model.eval()
            with torch.no_grad():
                loss, _, _, _ = calculate_loss(model, device, dataLoader, methode, sigma, min_value2, max_value2, batch_idx, original, noise_images)
                if methode == "n2noise":
                    denoised = model (noise_images)            
                elif methode == "n2score":
                    vector =  model(noise_images)
                    best_tv, best_sigma = evaluateSigma(noise_images, vector)
                    best_sigmas.append(best_sigma)
                    all_tvs.append(best_tv)
                    denoised = noise_images + best_sigma**2 * vector
                elif "n2self" in methode:
                    if "j-invariant" in methode:
                        denoised = Mask.n2self_jinv_recon(noise_images, model)
                    else:
                        denoised = model(noise_images)
                elif methode == "n2void" or "n2same" in methode:
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
            original_void = original
            loss, denoised, original, noise_images = calculate_loss(model, device, dataLoader, methode, sigma, min_value2, max_value2, batch_idx, original, noise_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #log Data
        psnr_original = Metric.calculate_psnr(original, denoised)
        denoised = (denoised-denoised.min())  / (denoised.max() - denoised.min())
        noise_images = (noise_images-noise_images.min())  / (noise_images.max() - noise_images.min())
        original = (original-original.min())  / (original.max() - original.min())
        psnr_batch = Metric.calculate_psnr(original, denoised)
        similarity_batch, diff_picture = Metric.calculate_similarity(original, denoised)
        loss_log.append(loss.item())
        psnr_orig_log.append(psnr_original.item())
        psnr_log.append(psnr_batch.item())
        sim_log.append(similarity_batch)


        if round(psnr_batch.item(),1) > bestPsnr + 0.5 or batch_idx == len(dataLoader)-1:# or similarity_batch > bestSim:
            if round(psnr_batch.item(),1) > bestPsnr:
                bestSim = similarity_batch
                bestPsnr = round(psnr_batch.item(),1)
                print(f"best model found with psnr: {bestPsnr}")
                model_save_path = os.path.join(store, "models", f"{round(bestPsnr, 1)}-{mode}-{epoch}-{batch_idx}.pth")
                if save_model:
                    torch.save(model.state_dict(), model_save_path)
                else:
                    f = open(model_save_path, "x")
                    f.close()
            if methode == "n2void" and mode == "train":
                original = original_void
                skip = int(denoised.shape[0] / batch) #64=ursprüngllichhe Batchgröße
                denoised = denoised[::skip] # zeige nur jedes 6. Bild an (im path wird aus einem bild 6 wenn die Batchhgröße = 64)
                original = original[::skip]
                noise_images = noise_images[::skip]
            #grid = make_grid(denoised, nrow=16, normalize=False) # Batch/number bilder im Raster
            comparison = torch.cat((original[:4], denoised[:4], noise_images[:4]), dim=0)
            grid = make_grid(comparison, nrow=4, normalize=False).cpu()
            if mode == "train":
                writer.add_image('Denoised Images Training', grid, global_step=epoch * len(dataLoader) + batch_idx)
            else:
                writer.add_image('Denoised Images Validation', grid, global_step=epoch * len(dataLoader) + batch_idx)
    """
    if (mode=="test" or mode =="validate") and (methode == "n2score"):
        for i in range(len(best_tv)):
            writer.add_scalar('Validation sigma', best_sigmas[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Validation tv', best_tv[i], epoch * len(dataLoader) + i)
    """
    show_tensor_as_picture
    return loss_log, psnr_log, sim_log, bestPsnr, psnr_orig_log

        

def main(argv):
    print("Starte Programm!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    methoden_liste = ["n2noise", "n2score", "n2self", "n2self j-invariant", "n2same", "n2same batch", "n2void"]
    #methoden_liste = ["n2void"]

    layout = {
        "Training vs Validation": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
            "PSNR": ["Multiline", ["PSNR/train", "PSNR/validation"]],
            "Similarity": ["Multiline", ["sim/train", "sim/validation"]],
        },
    }
    #writer = None
    sigma = 0.4
    sigma = 2
    save_model = False #save models as pth

    celeba_dir = 'dataset/celeba_dataset'

    
    transform_noise = transforms.Compose([
        #transforms.RandomResizedCrop((128,128)),
        transforms.CenterCrop((128,128)),
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
    
    for methode in methoden_liste:
        print(methode)
        if methode == "n2void":
            dataset = torch.utils.data.Subset(dataset, list(range(1056)))
            dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(128)))
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=False)
        if torch.cuda.device_count() == 1:
            model = N2N_Orig_Unet(3,3).to(device)
            #model = U_Net().to(device)
        else:
            #model = TestNet(3,3).to(device)
            if methode == "n2score" or methode == "n2void" or "batch" in methode:
                model = U_Net(batchNorm=True).to(device)
            else:
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
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer, min_value=-3.25, max_value=3.25, 
                                        min_value2=-5.67, max_value2=5.83, save_model=save_model)
            
            for i, loss_item in enumerate(loss):
                writer.add_scalar('Train Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Train PSNR_iteration [0,1]', psnr[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Train PSNR_iteration [-1,1]', psnr_orig[i], epoch * len(dataLoader) + i) #raw
                writer.add_scalar('Train Similarity_iteration', similarity[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Train PSNR [0,1] avg', Metric.avg_list(psnr), epoch)
            writer.add_scalar('Train PSNR [-1,1] avg', Metric.avg_list(psnr_orig), epoch) #raw
            
            high_psnr = max(psnr)
            high_sim = max(similarity)
            if epoch % 5 == 0 or round(max(psnr),1) > bestPsnr or epoch==max_Epochs-1:
                if round(max(psnr),1) > bestPsnr:
                    bestPsnr = round(max(psnr),1)
                model_save_path = os.path.join(store_path, "models", f"{epoch}-model.pth")
                if save_model:
                    torch.save(model.state_dict(), model_save_path)
                else:
                    f = open(model_save_path, "x")
                    f.close()
            
            print("Epochs highest PSNR: ", high_psnr)
            print("Epochs highest Sim: ", high_sim)
            
            if torch.cuda.device_count() == 1:
                continue
            
            
            #runing on Server
            loss_val, psnr_val, similarity_val, bestPsnr_val, psnr_orig_val = train(model, optimizer, device, dataLoader_validate, methode, sigma=sigma, mode="validate", 
                                                                                    store=store_path, epoch=epoch, bestPsnr=bestPsnr_val, writer = writer, min_value=-3.2, 
                                                                                    max_value=3.15, min_value2=-5.4, max_value2=5.32, save_model=save_model)
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