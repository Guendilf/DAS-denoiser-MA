import math
import os
import sys
import time
from pathlib import Path
#sys.path.insert(0, str(Path(__file__).resolve().parents[3])) #damit die Pfade auf dem Server richtig waren (copy past von PG)
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

import config

from models.N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self, U_Net_origi, U_Net, TestNet
from metric import Metric
from masks import Mask
from models.P_Unet import P_U_Net
from utils import *
from transformations import *
from loss import calculate_loss, noise2info

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



max_Iteration = 2
max_Epochs = 20
max_Predictions = 100 #für self2self um reconstruktion zu machen
torch.manual_seed(42)

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
    true_sigma_score = []
    bestPsnr = bestPsnr
    bestSim = -1
    n = torch.tensor([]).reshape(0, 3*128*128).to(device) #n like in n2info for estimate sigma
    loss_ex = 0
    loss_inv = 0
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
                loss, _, _, _, optional_tuples = calculate_loss(model, device, dataLoader, methode, sigma, true_noise_sigma, batch_idx, original, noise_images, augmentation, lambda_inv=lambda_inv, dropout_rate=dropout_rate)
                (_, _, _, est_sigma_opt) = optional_tuples
                if methode == "n2noise":
                    denoised = model (noise_images)            
                elif methode == "n2score":
                    true_sigma_score.append(true_noise_sigma)
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
                elif "self2self" in methode:
                    denoised = torch.ones_like(noise_images)
                    for i in range(max_Predictions):
                        _, denoised_tmp, _, _, flip = calculate_loss(model, device, dataLoader, methode, sigma, true_noise_sigma, batch_idx, original, noise_images, augmentation, lambda_inv=lambda_inv, dropout_rate=dropout_rate)
                        (lr, ud, _, _) = flip
                        #denoised_tmp = filp_lr_ud(denoised_tmp, lr, ud)
                        denoised = denoised + denoised_tmp
                    denoised = denoised / max_Predictions
                    denoised = (denoised+1)/2
                elif "n2info" in methode:
                    #TODO: normalisierung ist in der implementation da, aber ich habe es noch nicht im training gefunden

                    _, denoised, loss_inv_tmp, loss_ex_tmp = noise2info(noise_images, model, device, sigma_info)
                    loss_ex += loss_ex_tmp#+=
                    loss_inv += loss_inv_tmp#+=

                    n_partition = (denoised-noise_images).view(denoised.shape[0], -1) # (b, c*w*h)
                    n_partition = torch.sort(n_partition, dim=1).values
                    n = torch.cat((n, n_partition), dim=0)
                    #n=n_partition

                    if batch_idx == len(dataLoader) -1:
                        loss_inv = loss_inv / len(dataLoader)
                        est_sigma_opt = estimate_opt_sigma(noise_images, denoised, kmc=10, l_in=loss_inv, l_ex=loss_ex, n=n).item()
                        if est_sigma_opt < sigma_info:
                            sigma_info = est_sigma_opt
                    #best_sigmas.append(est_sigma_opt)

        else:
            model.train()
            #original, noise_images are only important if n2void
            original_void = original
            loss, denoised, original, noise_images, optional_tuples = calculate_loss(model, device, dataLoader, methode, sigma, true_noise_sigma, batch_idx, original, noise_images, augmentation, lambda_inv=lambda_inv, dropout_rate=dropout_rate, sigma_info=sigma_info)
            if "n2info" in methode and batch_idx == len(dataLoader):
                (_,_,sigma_info, est_sigma_opt) = optional_tuples
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
        #log sigma value for noise2info
        
        if "n2info" in methode and batch_idx == len(dataLoader)-1:
            #if mode=="train":
                #writer.add_scalar('Train estimated sigma', est_sigma_opt, epoch * len(dataLoader) + batch_idx)
            if mode=="validate":
                writer.add_scalar('Validate estimated sigma', est_sigma_opt, epoch)
            if mode=="test":
                writer.add_scalar('Test estimated sigma', est_sigma_opt, 1 * len(dataLoader) + batch_idx)
        """
    if (mode=="test" or mode =="validate") and ("n2info" in methode):
        if mode=="validate":
            for i in range(len(best_sigmas)):
                writer.add_scalar('Validate estimated sigma', best_sigmas[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Validation mean estimated', np.mean(best_sigmas), epoch )
        else: #mode=="test":
            for i in range(len(best_sigmas)):
                writer.add_scalar('Test estimated sigma', best_sigmas[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Test estimated sigma', np.mean(best_sigmas), epoch)
    """
    if (mode=="test" or mode =="validate") and ("n2score" in methode):
        ture_sigma_line = np.mean(true_sigma_score)
        if mode == "validate":
            writer.add_scalar('Validation true sigma', ture_sigma_line, epoch )
            writer.add_scalar('Validation sigma', np.mean(best_sigmas), epoch )
            writer.add_scalar('Validation tv', np.mean(best_tv), epoch)
            for i in range(len(best_sigmas)):
                writer.add_scalar('Validation all sigmas', best_sigmas[i], epoch * len(dataLoader) + i)
        else:
            writer.add_scalar('Test ture sigma', ture_sigma_line, epoch )
            writer.add_scalar('Test sigma', np.mean(best_sigmas), epoch )
            writer.add_scalar('Test tv', np.mean(best_tv), epoch)
    
    
    return loss_log, psnr_log, sim_log, bestPsnr, sigma_info



        

def main(argv):
    print("Starte Programm!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    methoden_liste = ["n2noise", "n2score", "n2self", "n2self j-invariant", "n2same batch", "n2info batch", "n2void"] #"self2self"
    methoden_liste = ["n2info batch"]

    layout = {
        "Training vs Validation": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
            "PSNR": ["Multiline", ["PSNR/train", "PSNR/validation"]],
            "Similarity": ["Multiline", ["sim/train", "sim/validation"]],
        },
    }
    #writer = None
    sigma = config.sigma
    sigma = config.sigmadb
    save_model = config.save_model #save models as pth

    celeba_dir = config.celeba_dir

    end_results = pd.DataFrame(columns=methoden_liste)

    
    transform_noise = transforms.Compose([
        #transforms.RandomResizedCrop((128,128)),
        transforms.CenterCrop((128,128)),
        #transforms.Resize((512,512)), #for self2self
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x:  x * 2 -1),
        ])
    print("lade Datensätze ...")
    dataset = datasets.CelebA(root=celeba_dir, split='train', download=False, transform=transform_noise)
    dataset = torch.utils.data.Subset(dataset, list(range(6400)))
    #dataset = torch.utils.data.Subset(dataset, list(range(2)))
    
    dataset_validate = datasets.CelebA(root=celeba_dir, split='valid', download=False, transform=transform_noise)
    dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(640)))
    #dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(1)))

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
            model = N2N_Orig_Unet(3,3).to(device) #default
            #model = P_U_Net(in_chanel=3, batchNorm=True, dropout=0.3).to(device)
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
        if "self2self" in methode:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        bestPsnr = -100
        bestPsnr_val = -100
        bestSim = -100
        bestSim_val = -100
        store_path = log_files()
        writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))
        writer.add_custom_scalars(layout)
        sigma_info = 1 #noise2Info starting pointt for estimated sigma

        for epoch in tqdm(range(max_Epochs)):
            
            loss, psnr, similarity, bestPsnr, sigma_info = train(model, optimizer, device, dataLoader, methode, sigma=sigma, mode="train", 
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer, save_model=save_model, 
                                        sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True)
            
            for i, loss_item in enumerate(loss):
                writer.add_scalar('Train Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Train PSNR_iteration [0,1]', psnr[i], epoch * len(dataLoader) + i)
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
            loss_val, psnr_val, similarity_val, bestPsnr_val, sigma_info = train(model, optimizer, device, dataLoader_validate, methode, sigma=sigma, mode="validate", 
                                                                    store=store_path, epoch=epoch, bestPsnr=bestPsnr_val, writer = writer, 
                                                                    save_model=save_model, sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True)

            for i, loss_item in enumerate(loss_val):
                writer.add_scalar('Validation Loss', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('Validation PSNR_iteration [0,1]', psnr_val[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Validation Similarity_iteration', similarity_val[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Validation PSNR [0,1] avg', Metric.avg_list(psnr_val), epoch)

            if max(psnr_val) > bestPsnr_val:
                bestPsnr_val = max(psnr_val)
            print("Loss: ", loss_val[-1])
            print("Epochs highest PSNR: ", max(psnr_val))
            print("Epochs highest Sim: ", max(similarity_val))
            writer.add_scalar("loss/train", Metric.avg_list(loss), epoch)
            writer.add_scalar("loss/validation", Metric.avg_list(loss_val), epoch)
            writer.add_scalar("PSNR/train", Metric.avg_list(psnr), epoch)
            writer.add_scalar("PSNR/validation", Metric.avg_list(psnr_val), epoch)
            writer.add_scalar("sim/train", Metric.avg_list(similarity), epoch)
            writer.add_scalar("sim/validation", Metric.avg_list(similarity_val), epoch)
        
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
        
        if torch.cuda.device_count() == 1:
            continue
        loss_test, psnr_test, similarity_test, _, _ = train(model, optimizer, device, dataLoader_test, methode, sigma=sigma, mode="test", 
                                                                    store=store_path, epoch=-1, bestPsnr=-1, writer = writer, 
                                                                    save_model=False, sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True)
        writer.add_scalar("PSNR Test", Metric.avg_list(psnr_test), 0)
        writer.add_scalar("Loss Test", Metric.avg_list(loss_test), 0)
        writer.add_scalar("Sim Test", Metric.avg_list(similarity_test), 0)

        end_results[methode] = [loss[-1], bestPsnr, bestSim, bestPsnr_val, bestSim_val, round(max(psnr_test),3), round(max(similarity_test),3)]

    end_results.index = ['Loss', 'PSNR Training', 'SIM Training', 'PSNR Validation', 'SIM Validation', 'PSNR Test', 'SIM Test']

    #show_logs(loss, psnr, value_loss, value_psnr, similarity)
    end_results.to_csv(os.path.join(store_path, "result_tabel.csv"))
    print(end_results)
    print(sigma_info)
    #show_pictures_from_dataset(dataset, model)
    writer.close()
    print("fertig\n")
    print(str(datetime.now().replace(microsecond=0)).replace(':', '-'))
    

if __name__ == "__main__":
    app.run(main)