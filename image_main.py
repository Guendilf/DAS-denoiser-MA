import math
import os
import sys
import time
from pathlib import Path
#sys.path.insert(0, str(Path(__file__).resolve().parents[3])) #damit die Pfade auf dem Server richtig waren (copy past von PG)
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

from models.N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self, U_Net_origi, U_Net, TestNet
from metric import Metric
from masks import Mask
from models.P_Unet import P_U_Net
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


from absl import app
from torch.utils.tensorboard import SummaryWriter
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\Code\DAS-denoiser-MA\runs"      #PC
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\runs"                           #vom Server



max_Iteration = 2
max_Epochs = 20
max_Predictions = 30 #für self2self um reconstruktion zu machen

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

"""
TRAIN FUNKTIONS
"""

def train(model, optimizer, device, dataLoader, methode, sigma, mode, store, epoch, bestPsnr, writer, save_model, sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True):
    #mit lossfunktion aus "N2N(noiseToNoise)"
    #kabel ist gesplitet und ein Teil (src) wird im Model behandelt und der andere (target) soll verglichen werden
    loss_log = []
    psnr_log = []
    sim_log = []
    best_sigmas = []    #only for Score in validation + test
    all_tvs = []     #only for Score in validation + test
    bestPsnr = bestPsnr
    bestSim = -1
    for batch_idx, (original, label) in enumerate(tqdm(dataLoader)):
        original = original.to(device)
        batch = original.shape[0]
        #noise_images = add_norm_noise(original, sigma, min_value, max_value, a=-1, b=1)
        noise_images, true_noise_sigma = add_noise_snr(original, snr_db=sigma)
        noise_images = noise_images.to(device)
        #get specific values for training and validation
        if mode=="test" or mode =="validate":
            model.eval()
            with torch.no_grad():
                loss, _, _, _, _ = calculate_loss(model, device, dataLoader, methode, sigma, true_noise_sigma, batch_idx, original, noise_images, augmentation, lambda_inv=lambda_inv, dropout_rate=dropout_rate)
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
                elif "n2void" in methode or "n2same" in methode:
                    #calculate mean and std for each Image in batch in every chanal
                    mean = noise_images.mean(dim=[0,2,3])
                    std = noise_images.std(dim=[0,2,3])
                    noise_images = (noise_images - mean[None, :, None, None]) / std[None, :, None, None]
                    denoised = model(noise_images)
                elif "self2slef" in methode:
                    denoised = torch.ones_like(noise_images)
                    for i in range(max_Predictions):
                        _, denoised_tmp, _, _, flip = calculate_loss(model, device, dataLoader, methode, sigma, true_noise_sigma, batch_idx, original, noise_images, augmentation, lambda_inv=lambda_inv, dropout_rate=dropout_rate)
                        (lr, ud, _) = flip
                        denoised_tmp = filp_lr_ud(denoised_tmp, lr, ud)
                        denoised = denoised + denoised_tmp
                elif "n2info" in methode:
                    #TODO: normalisierung ist in der implementation da, aber ich habe es noch nicht im training gefunden
                    denoised = model(noise_images)

        else:
            model.train()
            #original, noise_images are only important if n2void
            original_void = original
            loss, denoised, original, noise_images, optional_tuples = calculate_loss(model, device, dataLoader, methode, sigma, true_noise_sigma, batch_idx, original, noise_images, augmentation, lambda_inv=lambda_inv, dropout_rate=dropout_rate, sigma_info=sigma_info)
            if "n2info" in methode and batch_idx == len(dataLoader):
                (_,_,sigma_info) = optional_tuples
                #print(sigma_info)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #log Data
        denoised = (denoised-denoised.min())  / (denoised.max() - denoised.min())
        noise_images = (noise_images-noise_images.min())  / (noise_images.max() - noise_images.min())
        original = (original-original.min())  / (original.max() - original.min())
        psnr_batch = Metric.calculate_psnr(original, denoised)
        similarity_batch, diff_picture = Metric.calculate_similarity(original, denoised)
        loss_log.append(loss.item())
        psnr_log.append(psnr_batch.item())
        sim_log.append(similarity_batch)

        #save model + picture
        bestPsnr = saveModel_pictureComparison(model, len(dataLoader), methode, mode, store, epoch, bestPsnr, writer, save_model, batch_idx, original, batch, noise_images, denoised, psnr_batch)
    """
    if (mode=="test" or mode =="validate") and (methode == "n2score"):
        for i in range(len(best_tv)):
            writer.add_scalar('Validation sigma', best_sigmas[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Validation tv', best_tv[i], epoch * len(dataLoader) + i)
    """
    return loss_log, psnr_log, sim_log, bestPsnr



        

def main(argv):
    print("Starte Programm!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    methoden_liste = ["n2noise", "n2score", "n2self", "n2self j-invariant", "n2same", "n2same batch", "n2info", "self2self", "n2void"]
    #methoden_liste = ["n2info"]

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

    dataset_test = datasets.CelebA(root=celeba_dir, split='test', download=False, transform=transform_noise)
    dataset_test = torch.utils.data.Subset(dataset_test, list(range(640)))
    
    print(f"Using {device} device")
    
    for methode in methoden_liste:
        print(methode)
        if methode == "n2void":
            dataset = torch.utils.data.Subset(dataset, list(range(1056)))
            dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(128)))
            dataset_test = torch.utils.data.Subset(dataset_test, list(range(128)))
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
        if torch.cuda.device_count() == 1:
            model = N2N_Orig_Unet(3,3).to(device)
            #model = P_U_Net(in_chanel=3, batchNorm=False, dropout=0.3).to(device)
            #model = U_Net().to(device)
        else:
            #model = TestNet(3,3).to(device)
            if methode == "n2score" or methode == "n2void" or "batch" in methode:
                model = U_Net(batchNorm=True).to(device)
            #model = Cut2Self(mask).to(device)
            elif "self2self" in methode:
                model = P_U_Net(in_chanel=3, batchNorm=False, dropout=0.3).to(device)
            else:
                model = U_Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        bestPsnr = -100
        bestPsnr_val = -100
        store_path = log_files()
        writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))
        writer.add_custom_scalars(layout)
        sigma_info = 1 #noise2Info starting pointt for estimated sigma

        for epoch in tqdm(range(max_Epochs)):
            
            loss, psnr, similarity, bestPsnr = train(model, optimizer, device, dataLoader, methode, sigma=sigma, mode="train", 
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer, save_model=save_model, 
                                        sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True)
            
            for i, loss_item in enumerate(loss):
                writer.add_scalar('Train Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Train PSNR_iteration [0,1]', psnr[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Train Similarity_iteration', similarity[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Train PSNR [0,1] avg', Metric.avg_list(psnr), epoch)
            
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
            
            if math.isnan(loss[-1]):
                model_save_path = os.path.join(store_path, "models", "NaN.txt")
                f = open(model_save_path, "x")
                f.close()
                break
            if torch.cuda.device_count() == 1:
                continue
            
            
            #runing on Server
            loss_val, psnr_val, similarity_val, bestPsnr_val = train(model, optimizer, device, dataLoader_validate, methode, sigma=sigma, mode="validate", 
                                                                    store=store_path, epoch=epoch, bestPsnr=bestPsnr_val, writer = writer, 
                                                                    save_model=save_model, sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True)
            high_psnr = -100
            high_sim = -100
            for i, loss_item in enumerate(loss_val):
                writer.add_scalar('Validation Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Validation PSNR_iteration [0,1]', psnr_val[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Validation Similarity_iteration', similarity_val[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Validation PSNR [0,1] avg', Metric.avg_list(psnr_val), epoch)
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
            writer.add_scalar("sim/train", Metric.avg_list(similarity), epoch)
            writer.add_scalar("sim/validation", Metric.avg_list(similarity_val), epoch)
        
        if torch.cuda.device_count() == 1:
            continue
        loss_test, psnr_test, similarity_test, _ = train(model, optimizer, device, dataLoader_test, methode, sigma=sigma, mode="test", 
                                                                    store=store_path, epoch=-1, bestPsnr=-1, writer = writer, 
                                                                    save_model=False, sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True)
        writer.add_scalar("PSNR Test", Metric.avg_list(psnr_test), 0)
        writer.add_scalar("Loss Test", Metric.avg_list(loss_test), 0)
        writer.add_scalar("Sim Test", Metric.avg_list(similarity_test), 0)

    #show_logs(loss, psnr, value_loss, value_psnr, similarity)

    print(loss)
    print(psnr)
    print(similarity)
    #show_pictures_from_dataset(dataset, model)
    writer.close()
    print("fertig\n")
    print(str(datetime.now().replace(microsecond=0)).replace(':', '-'))
    

if __name__ == "__main__":
    app.run(main)