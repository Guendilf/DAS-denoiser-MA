import math
import os
import sys
import time
from pathlib import Path
#sys.path.insert(0, str(Path(__file__).resolve().parents[3])) #damit die Pfade auf dem Server richtig waren (copy past von PG)
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

import config_test as config
#import config as config

from models.N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self, U_Net_origi, U_Net, TestNet
from metric import Metric
from masks import Mask
from models.P_Unet import P_U_Net
from utils import *
from transformations import *
from loss import calculate_loss, n2same,  n2info

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
        if "n2noise" in methode:
            noise_images2 = n2n_create_2_input(device, methode, original, noise_images)
            #if "test" in methode:
                #noise_images2 = n2n_create_2_input(device, methode, original, original)
            #noise_images2 = torch.clip(noise_images2, 0,1.0)
        else:
            noise_images2 = None
        #noise_images = torch.clip(noise_images, 0,1.0)

        if mode=="test" or mode =="validate":
            model.eval()
            with torch.no_grad():
                loss, _, _, _, optional_tuples = calculate_loss(model, device, dataLoader, methode, true_noise_sigma, batch_idx, original, noise_images, noise_images2, augmentation, dropout_rate=dropout_rate, radius=3)
                (_, _, _, est_sigma_opt) = optional_tuples
                if "n2noise" in methode:
                    denoised = model (noise_images)            
                elif "n2score"  in methode:
                    true_sigma_score.append(true_noise_sigma)
                    vector =  model(noise_images)
                    best_tv, best_sigma = evaluateSigma(noise_images, vector)
                    best_sigmas.append(best_sigma)
                    all_tvs.append(best_tv)
                    denoised = noise_images + best_sigma**2 * vector
                elif "n2self" in methode:
                    #if "j-invariant" in methode:
                    if "j-invariant" in config.methodes[methode]['erweiterung']:
                        denoised = Mask.n2self_jinv_recon(noise_images, model)
                    else:
                        denoised = model(noise_images)
                elif "n2same" in methode:
                    denoised = model(noise_images)
                elif "n2void" in methode:
                    #calculate mean and std for each Image in batch in every chanal
                    #mean = noise_images.mean(dim=[0,2,3])
                    #std = noise_images.std(dim=[0,2,3])
                    #noise_images = (noise_images - mean[None, :, None, None]) / std[None, :, None, None]
                    noise_images = (noise_images - noise_images.mean(dim=(1, 2), keepdim=True)) / noise_images.std(dim=(1, 2), keepdim=True)
                    denoised = model(noise_images)
                elif "s2self" in methode:
                    denoised = torch.ones_like(noise_images)
                    for i in range(max_Predictions):
                        _, denoised_tmp, _, _, flip = calculate_loss(model, device, dataLoader, methode, true_noise_sigma, batch_idx, original, noise_images, noise_images2, augmentation, dropout_rate=dropout_rate, radius=3)
                        (lr, ud, _, _) = flip
                        #denoised_tmp = filp_lr_ud(denoised_tmp, lr, ud)
                        denoised = denoised + denoised_tmp
                    denoised = denoised / max_Predictions
                    denoised = (denoised+1)/2
                elif "n2info" in methode:
                    #TODO: normalisierung ist in der implementation da, aber ich habe es noch nicht im training gefunden
                    """
                    !Old version
                    if batch_idx == 0:
                        est_sigma_opt = estimate_opt_sigma_new(noise_images, dataLoader, model, device, sigma_info).item()

                    _, denoised, loss_inv_tmp, loss_ex_tmp, _ = noise2info(noise_images, model, device, sigma_info)
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
                    best_sigmas.append(est_sigma_opt)
                    """
                    
                    if mode=="test":
                        denoised = model(noise_images)
                    else:
                        #loss, denoised, loss_rec, loss_inv, marked_pixel = n2same(noise_images, device, model, lambda_inv)
                        loss, denoised, loss_rec, loss_inv, marked_pixel = n2info(noise_images, model, device, sigma_info)
                        all_marked += marked_pixel
                        lex += loss_rec
                        lin += loss_inv
                        n_partition = (denoised-noise_images).view(denoised.shape[0], -1) # (b, c*w*h)
                        n_partition = torch.sort(n_partition, dim=1).values #descending=False
                        n = torch.cat((n, n_partition), dim=0)
                        if batch_idx == len(dataLoader)-1:
                            e_l = 0
                            for i in range(config.methodes['n2info']['predictions']): #kmc
                                #to big for torch.multinomial if all pictures from validation should be used
                                #samples = torch.tensor(torch.multinomial(n.view(-1), n.shape[1], replacement=True))#.view(1, n.shape[1])
                                #samples = torch.sort(samples).values
                                samples = np.sort(np.random.choice((n.cpu()).reshape(-1),[1, n.shape[1]])) #(1,49152)
                                e_l += torch.mean((n-torch.from_numpy(samples).to(device))**2)
                            lex = lex / (len(dataLoader) * denoised.shape[0])
                            lin = lin / all_marked
                            e_l = e_l / config.methodes['n2info']['predictions']
                            #estimated_sigma = (lin)**0.5 + (lin + lex-e_l)**0.5 #inplementation from original github of noise2info
                            m = len(dataLoader) * denoised.shape[0] *3*128*128 #TODO: is m right?
                            estimated_sigma = lex + (lex**2 + m *(lin-e_l))**0.5/m #from paper
                            print('new sigma_loss is ', estimated_sigma)
                            if 0 < estimated_sigma < sigma_info:
                                sigma_info = float(estimated_sigma)
                                print('sigma_loss updated to ', estimated_sigma)
                            writer.add_scalar('estimated sigma', estimated_sigma, epoch)
                            writer.add_scalar('lex', lex, epoch)
                            writer.add_scalar('lin', lin, epoch)
                            writer.add_scalar('e_l', e_l, epoch)
                        
        else:
            model.train()
            #original, noise_images are only important if n2void
            loss, denoised, original, noise_images, optional_tuples = calculate_loss(model, device, dataLoader, methode, true_noise_sigma, batch_idx, original, noise_images, noise_images2, augmentation, dropout_rate=dropout_rate, sigma_info=sigma_info, radius=3)# TODO:, radius=config.methodes[methode]['radius'])
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
    #log sigma value for noise2info
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
    """
    #log sigma value for n2score
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
    
    
    return loss_log, psnr_log, sim_log, bestPsnr, sigma_info, original_psnr_log



        

def main(argv):
    print("Starte Programm!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    methoden_liste = ["n2noise 1_input", "n2noise 2_input", "n2score", "n2self", "n2self j-invariant", "n2same batch", "n2info batch", "n2void"] #"self2self"
    #methoden_liste = ["n2void"]

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

    #end_results = pd.DataFrame(columns=methoden_liste)
    end_results = pd.DataFrame(columns=config.methodes.keys())

    
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

    methoden_liste = ["n2same"]
    #for methode in methoden_liste:
        #method_params = config.methodes[methode]
    for methode, method_params in config.methodes.items():

        #create to folders for loging details
        store_path = Path(os.path.join(store_path_root, methode))
        store_path.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "tensorboard"))
        tmp.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "models"))
        tmp.mkdir(parents=True, exist_ok=True)

        print(methode)
        if methode == "n2void":
            dataset = torch.utils.data.Subset(dataset_all, list(range(1056)))
            dataset_validate = torch.utils.data.Subset(dataset_validate_all, list(range(128)))
            dataset_test = torch.utils.data.Subset(dataset_test_all, list(range(128)))
        else:
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
            if "n2same" in methode or "n2info" in methode:
                model = U_Net(in_chanel=3, first_out_chanel=96, batchNorm=method_params['batchNorm']).to(device)
            elif "s2self" in methode:
                model = P_U_Net(in_chanel=3, batchNorm=method_params['batchNorm'], dropout=method_params['dropout']).to(device)
            else:
                model = U_Net(in_chanel=3, batchNorm=method_params['batchNorm']).to(device)
        #configAtr = getattr(config, methode) #config.methode wobei methode ein string ist
        optimizer = torch.optim.Adam(model.parameters(), lr=method_params['lr'])
        if method_params['sheduler']:
            lr_lambda = get_lr_lambda(method_params['lr'], method_params['changeLR_steps'], method_params['changeLR_rate'])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
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