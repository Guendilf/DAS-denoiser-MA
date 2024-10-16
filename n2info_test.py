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
max_Epochs = 1
max_Predictions = 100 #für self2self um reconstruktion zu machen
torch.manual_seed(42)

def get_lr_lambda(initial_lr, step_size, lr_decrement):
    def lr_lambda(step):
        return max(0.0, initial_lr - (step // step_size) * lr_decrement / initial_lr)
    return lr_lambda


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

def old_estimate_opt_sigma(noise_images, denoised, kmc, l_in, l_ex, n):
    """
    used for Noise2Info to determin a better suited sigma for loss calculation
    Args:
        moodel: the used torch model
        noise_images (tensor): images with noise with sahpe: (b,c,w,h)
        samples: samples for Monte Carloo integration
        n: as in paper - it's an argument, because I have multiple batches that must be put together first
    Returns:
        best sigma value
    """
    e_l = 0
    #n = (denoised-noise_images).view(denoised.shape[0], -1) # (b, c*w*h)
    #n = torch.sort(n, dim=1).values #sort every batch
    m = denoised.shape[1]*denoised.shape[2]*denoised.shape[3]
    all_pixels = n.shape[0]*m
    for i in range(kmc):# TODO: checken ob richhtiger wert aus k_mc
        # sample uniform between 0 and max(pixel count in images) exacly "samples" pixels
        indices = torch.randperm(all_pixels)[:m]
        # transform n into vector view and extract the sampled values
        sampled_values = n.view(-1)[indices]
        sorted_values = torch.sort(sampled_values).values #result from sort: [values, indices]

        
        e_l += torch.mean((n - sorted_values) ** 2)
    e_l = e_l / kmc
    #Equation 6
    sigma = l_ex + (l_ex**2 + all_pixels*(l_in-e_l)).sqrt()/all_pixels#TODO: e_l ist sehr groß und l_in sehr klein -> NaN weegen wurzel
    return sigma

def paper_estimate_opt_sigma_new(noise_images, dataLoader, model, device, sigma_info):
    """
    used for Noise2Info to determin a better suited sigma for loss calculation
    Args:
        noise_images (tensor): images used to extract shape (b,c,w,h)
        dataLoader: to iterate over all pictures in the validation dataset
        moodel: the used torch model
        device: save tensors on cpu or gpu
        sigma_info: used for calculation of losses (old best sigma value)
    Returns:
        estimated sigma value
    """
    loss_in = 0
    loss_ex = 0
    e_l = 0
    all_marked_points = 0
    batchsize = noise_images.shape[0]
    dimension = noise_images.shape[1]*noise_images.shape[2]*noise_images.shape[3]
    all_pixels = len(dataLoader)*batchsize*dimension
    n = torch.zeros(len(dataLoader)*batchsize, dimension).to(device)
    for batch_idx, (original, label) in enumerate((dataLoader)):
        if config.useSigma:
            noise_images = add_norm_noise(original, config.sigma, a=-1, b=1, norm=False)
            true_noise_sigma = config.sigma
        else:
            noise_images, true_noise_sigma = add_noise_snr(original, snr_db=config.sigmadb)
        noise_images = noise_images.to(device)
        _, denoised, loss_inv_tmp, loss_ex_tmp, marked_points = noise2info(noise_images, model, device, sigma_info)
        loss_in += loss_inv_tmp
        loss_ex += loss_ex_tmp
        all_marked_points += marked_points
        #save every picture as vector in corresponding row of n
        n[batch_idx*batchsize : batch_idx*batchsize + batchsize, :] = denoised.view(batchsize, -1)
    loss_ex = loss_ex / all_marked_points
    loss_in = loss_in / len(dataLoader)
    for i in range(config.methodes['n2info']['predictions']):
        # sample uniform between 0 and max(pixel count in all images) exacly "dimension" pixels
        indices = torch.randperm(all_pixels)[:dimension] #dimension = m in paper
        # transform n into vector view and extract the sampled values
        sampled_values = n.view(-1)[indices]
        sorted_values = torch.sort(sampled_values).values #result from sort: [values, indices]

        
        e_l += torch.mean((n - sorted_values) ** 2)
    e_l = e_l / config.methodes['n2info']['predictions']
    #estimated_sigma= loss_ex**0.5 + (loss_ex + loss_in - e_l)**0.5 # version from github https://github.com/dominatorX/Noise2Info-code/blob/master/network_keras.py#L106
    estimated_sigma = loss_ex + (loss_ex**2 + dimension*(loss_in-e_l)).sqrt()/dimension #version from paper
    return estimated_sigma

def git_paper_estimate_opt_sigma_new(noise_images, dataLoader, model, device, sigma_info):
    """
    used for Noise2Info to determin a better suited sigma for loss calculation
    Args:
        noise_images (tensor): images used to extract shape (b,c,w,h)
        dataLoader: to iterate over all pictures in the validation dataset
        moodel: the used torch model
        device: save tensors on cpu or gpu
        sigma_info: used for calculation of losses (old best sigma value)
    Returns:
        estimated sigma value
    """
    loss_in = 0
    loss_ex = 0
    e_l = 0
    all_marked_points = 0
    batchsize = noise_images.shape[0]
    dimension = noise_images.shape[1]*noise_images.shape[2]*noise_images.shape[3]
    all_pixels = len(dataLoader)*batchsize*dimension
    n = torch.zeros(len(dataLoader)*batchsize, dimension).to(device)
    for batch_idx, (original, label) in enumerate((dataLoader)):
        if config.useSigma:
            noise_images = add_norm_noise(original, config.sigma, a=-1, b=1, norm=False)
            true_noise_sigma = config.sigma
        else:
            noise_images, true_noise_sigma = add_noise_snr(original, snr_db=config.sigmadb)
        noise_images = noise_images.to(device)
        _, denoised, loss_inv_tmp, loss_ex_tmp, marked_points = noise2info(noise_images, model, device, sigma_info)
        loss_in += loss_inv_tmp
        loss_ex += loss_ex_tmp
        all_marked_points += marked_points
        #save every picture as vector in corresponding row of n
        n[batch_idx*batchsize : batch_idx*batchsize + batchsize, :] = denoised.view(batchsize, -1)
    loss_ex = loss_ex / all_marked_points
    loss_in = loss_in / len(dataLoader)
    for i in range(config.methodes['n2info']['predictions']):
        # sample uniform between 0 and max(pixel count in all images) exacly "dimension" pixels
        indices = torch.randperm(all_pixels)[:dimension] #dimension = m in paper
        # transform n into vector view and extract the sampled values
        sampled_values = n.view(-1)[indices]
        sorted_values = torch.sort(sampled_values).values #result from sort: [values, indices]

        
        e_l += torch.mean((n - sorted_values) ** 2)
    e_l = e_l / config.methodes['n2info']['predictions']
    estimated_sigma= loss_ex**0.5 + (loss_ex + loss_in - e_l)**0.5 # version from github https://github.com/dominatorX/Noise2Info-code/blob/master/network_keras.py#L106
    #estimated_sigma = loss_ex + (loss_ex**2 + dimension*(loss_in-e_l)).sqrt()/dimension #version from paper
    return estimated_sigma

"""
TRAIN FUNKTIONS
"""

def train(model, optimizer, scheduler, device, dataLoader, methode, sigma, mode, store, epoch, bestPsnr, writer, save_model, sigma_info, dropout_rate=0.3, lambda_inv=2, augmentation=True):
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
        if config.useSigma:
            noise_images = add_norm_noise(original, sigma, a=-1, b=1, norm=False)
            true_noise_sigma = sigma
        else:
            noise_images, true_noise_sigma = add_noise_snr(original, snr_db=sigma)
        noise_images = noise_images.to(device)
        #get specific values for training and validation
        if mode=="test" or mode =="validate":
            model.eval()
            with torch.no_grad():
                loss, _, _, _, optional_tuples = calculate_loss(model, device, dataLoader, methode, sigma, true_noise_sigma, batch_idx, original, noise_images, augmentation, lambda_inv=lambda_inv, dropout_rate=dropout_rate)
                (_, _, _, est_sigma_opt) = optional_tuples
                
                if lambda_inv == 0:
                #TODO: normalisierung ist in der implementation da, aber ich habe es noch nicht im training gefunden
                    
                    _, denoised, loss_inv_tmp, loss_ex_tmp, _ = noise2info(noise_images, model, device, sigma_info)
                    loss_ex += loss_ex_tmp#+=
                    loss_inv += loss_inv_tmp#+=

                    n_partition = (denoised-noise_images).view(denoised.shape[0], -1) # (b, c*w*h)
                    n_partition = torch.sort(n_partition, dim=1).values
                    n = torch.cat((n, n_partition), dim=0)
                    #n=n_partition


                    if batch_idx == len(dataLoader) -1:
                        loss_inv = loss_inv / len(dataLoader)
                        est_sigma_opt = old_estimate_opt_sigma(noise_images, denoised, kmc=100, l_in=loss_inv, l_ex=loss_ex, n=n).item()

                    if est_sigma_opt < sigma_info:
                        sigma_info = est_sigma_opt
                    best_sigmas.append(est_sigma_opt)
                elif lambda_inv == 1:
                #TODO: normalisierung ist in der implementation da, aber ich habe es noch nicht im training gefunden
                    
                    _, denoised, loss_inv_tmp, loss_ex_tmp, _ = noise2info(noise_images, model, device, sigma_info)
                    loss_ex = loss_ex_tmp#+=
                    loss_inv = loss_inv_tmp#+=

                    n_partition = (denoised-noise_images).view(denoised.shape[0], -1) # (b, c*w*h)
                    n_partition = torch.sort(n_partition, dim=1).values
                    #n = torch.cat((n, n_partition), dim=0)
                    n=n_partition


                    #if batch_idx == len(dataLoader) -1:
                        #loss_inv = loss_inv / len(dataLoader)
                    est_sigma_opt = old_estimate_opt_sigma(noise_images, denoised, kmc=10, l_in=loss_inv, l_ex=loss_ex, n=n).item()

                    if est_sigma_opt < sigma_info:
                        sigma_info = est_sigma_opt
                    best_sigmas.append(est_sigma_opt)
                elif lambda_inv == 2:
                #TODO: normalisierung ist in der implementation da, aber ich habe es noch nicht im training gefunden
                    if batch_idx == 0:
                        est_sigma_opt = paper_estimate_opt_sigma_new(noise_images, dataLoader, model, device, sigma_info).item()
                        if est_sigma_opt < sigma_info:
                            sigma_info = est_sigma_opt
                    _, denoised, loss_inv_tmp, loss_ex_tmp, _ = noise2info(noise_images, model, device, sigma_info)
                    
                    best_sigmas.append(est_sigma_opt)
                elif lambda_inv == 3:
                #TODO: normalisierung ist in der implementation da, aber ich habe es noch nicht im training gefunden
                    if batch_idx == 0:
                        est_sigma_opt = git_paper_estimate_opt_sigma_new(noise_images, dataLoader, model, device, sigma_info).item()
                        if est_sigma_opt < sigma_info:
                            sigma_info = est_sigma_opt
                    _, denoised, loss_inv_tmp, loss_ex_tmp, _ = noise2info(noise_images, model, device, sigma_info)
                    
                    best_sigmas.append(est_sigma_opt)

        else:
            model.train()
            #original, noise_images are only important if n2void
            original_void = original
            loss, denoised, original, noise_images, optional_tuples = calculate_loss(model, device, dataLoader, methode, sigma, true_noise_sigma, batch_idx, original, noise_images, augmentation, lambda_inv=lambda_inv, dropout_rate=dropout_rate, sigma_info=sigma_info)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if config.methodes[methode]['sheduler']:
                scheduler.step()

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
        """
        if "n2info" in methode and batch_idx == len(dataLoader)-1:
            #if mode=="train":
                #writer.add_scalar('Train estimated sigma', est_sigma_opt, epoch * len(dataLoader) + batch_idx)
            if mode=="validate":
                writer.add_scalar('Validate estimated sigma', est_sigma_opt, epoch)
            #else: #mode=="test":
                #writer.add_scalar('Test estimated sigma', est_sigma_opt, 1 * len(dataLoader) + batch_idx)
        """
    if (mode=="test" or mode =="validate") and ("n2info" in methode):
        if mode=="validate":
            for i in range(len(best_sigmas)):
                writer.add_scalar('Validate estimated sigma', best_sigmas[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Validation mean estimated', np.mean(best_sigmas), epoch )
        if mode=="test":
            for i in range(len(best_sigmas)):
                writer.add_scalar('Test estimated sigma', best_sigmas[i], epoch * len(dataLoader) + i)
            writer.add_scalar('Test estimated sigma', np.mean(best_sigmas), epoch)
    
    
    
    return loss_log, psnr_log, sim_log, bestPsnr, sigma_info



        

def main(argv):
    print("Starte Programm!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    methoden_liste = ["n2noise 1_input", "n2noise 2_input", "n2score", "n2self", "n2self j-invariant", "n2same batch", "n2info batch", "n2void"] #"self2self"
    methoden_liste = ["n2noise 1_input", "n2noise 2_input"]
    lambda_invs= [0,1,2,3]
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

    end_results = pd.DataFrame(columns=lambda_invs)

    
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
    #dataset_test = torch.utils.data.Subset(dataset_test, list(range(1)))
    
    print(f"Using {device} device")
    
    #for methode in methoden_liste:
    #for methode, method_params in config.methodes.items():
    for lambda_values in lambda_invs:
        methode = 'n2info'
        print(methode)
        
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
        if torch.cuda.device_count() == 1:
            model = N2N_Orig_Unet(3,3).to(device) #default
           
        else:
            #if "n2same" in methode:
                #model = U_Net(first_out_chanel=96, batchNorm=method_params['batchNorm']).to(device)  
            model = U_Net(batchNorm=config.methodes['n2info']['batchNorm']).to(device)
        #configAtr = getattr(config, methode) #config.methode wobei methode ein string ist
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.methodes['n2info']['lr'])
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
        store_path = log_files()
        writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))
        writer.add_custom_scalars(layout)
        sigma_info = 1 #noise2Info starting pointt for estimated sigma

        for epoch in tqdm(range(max_Epochs)):
            
            loss, psnr, similarity, bestPsnr, sigma_info = train(model, optimizer, scheduler, device, dataLoader, methode, sigma=sigma, mode="train", 
                                        store=store_path, epoch=epoch, bestPsnr=bestPsnr, writer = writer, save_model=save_model, 
                                        sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=lambda_values, augmentation=True)
            
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
            loss_val, psnr_val, similarity_val, bestPsnr_val, sigma_info = train(model, optimizer, scheduler, device, dataLoader_validate, methode, sigma=sigma, mode="validate", 
                                                                    store=store_path, epoch=epoch, bestPsnr=bestPsnr_val, writer = writer, 
                                                                    save_model=save_model, sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=lambda_values, augmentation=True)

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
        
        if torch.cuda.device_count() == 1:
            continue
        loss_test, psnr_test, similarity_test, _, _ = train(model, optimizer, scheduler, device, dataLoader_test, methode, sigma=sigma, mode="test", 
                                                                    store=store_path, epoch=-1, bestPsnr=-1, writer = writer, 
                                                                    save_model=False, sigma_info=sigma_info, dropout_rate=0.3, lambda_inv=lambda_values, augmentation=True)
        writer.add_scalar("PSNR Test", Metric.avg_list(psnr_test), 0)
        writer.add_scalar("Loss Test", Metric.avg_list(loss_test), 0)
        writer.add_scalar("Sim Test", Metric.avg_list(similarity_test), 0)

        end_results[methode] = [loss[-1], bestPsnr, avg_train_psnr[-1], round(bestSim,3), round(avg_train_sim[-1],3),
                                bestPsnr_val, avg_val_psnr[-1], round(bestSim_val,3), round(avg_val_sim[-1],3),
                                round(max(psnr_test),3), round(max(similarity_test),3)]

    end_results.index = ['Loss', 'Max PSNR Training', 'Avg PSNR last Training', 'SIM Training', 'Avg SIM last Training',
                         'PSNR Validation', 'Avg PSNR last Training', 'SIM Validation', 'Avg SIM last Validation',
                         'PSNR Test', 'SIM Test']

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