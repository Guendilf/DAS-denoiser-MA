from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil

import torch
#import config
from loss import n2info


def show_pictures_from_dataset (dataset, model=None, generation=0):
    if model:
        with torch.no_grad():
            device = next(model.parameters()).device  
            denoised = model(dataset.to(device)).cpu()
            dataset = dataset.cpu()
        plt.figure(generation/100)
        plt.title( 'Abbildung ' +str(generation) )
        for i in range(5): #5 rows, 4 colums, dataset = src
        # Originalbild
            plt.subplot(5, 4, i * 4 + 1)
            image = dataset[i]
            plt.imshow(image.permute(1, 2, 0))
            plt.axis('off')
            if i == 0:
                plt.title('Original')

            # Eingangsbilder
            input1 = add_gaus_noise(image, 0.5, 0.1**0.5)
            input2 = add_gaus_noise(image, 0.6, 0.4**0.5)
            plt.subplot(5, 4, i * 4 + 2)
            plt.imshow(input1.permute(1, 2, 0))
            plt.axis('off')
            if i == 0:
                plt.title('Noise 1')
            
            plt.subplot(5, 4, i * 4 + 3)
            plt.imshow(input2.permute(1, 2, 0))
            plt.axis('off')
            if i == 0:
                plt.title('Noise 2')

            # Ergebnis des Netzes
            plt.subplot(5, 4, i * 4 + 4)
            plt.imshow(denoised[i].permute(1, 2, 0))
            plt.axis('off')
            if i == 0:
                plt.title('Denoised')
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 10))
        for i in range(9):
            image, _ = dataset[i]
            #image = add_gaus_noise(image, 0.5, 0.1**0.5)
            plt.subplot(3, 3, i + 1)
            plt.imshow(image.permute(1, 2, 0))
            plt.axis('off')
        plt.show()




def show_logs(loss, psnr, value_loss, value_psnr, similarity):
    x_axis = np.arange(len(loss))
    #plt.plot(loss, color='blue')
    #plt.plot(psnr, color='red')
    plt.show()
    fig, ax1 = plt.subplots()
    fig.suptitle('Trainingsverlauf')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('loss')
    ax1.set_yscale('log')
    ax1.plot(x_axis, loss, color='blue', label='training loss')
    #ax1.plot(x_axis, value_loss, color='red', label='value loss')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('psnr (dp)')  # we already handled the x-label with ax1
    ax2.plot(x_axis, psnr, color='green', label='training psnr')
    #ax2.plot(x_axis, value_psnr, color='yellow', label='value_psnr')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()


    figure3, ax3 = plt.subplots()
    figure3.suptitle('Validierung auf Test Daten')
    ax3.set_xlabel('Epochs/10')
    ax3.plot(value_loss, color='blue', label='value loss')
    ax3.set_yscale('log')
    ax3.set_ylabel('loss')

    ax4 = ax3.twinx()
    ax4.plot(value_psnr, color='red', label='value psnr')
    ax4.set_ylabel('psnr (dp)')
    plt.legend()

    figure5, ax5 = plt.subplots()
    figure5.suptitle('Similarity')
    ax5.set_xlabel('Epochs')
    ax5.plot(similarity, color='blue', label='similarity while training')

    plt.show()

def log_files():
    current_path = Path(os.getcwd())
    store_path = Path(os.path.join(current_path, "runs", f"run-{str(datetime.now().replace(microsecond=0)).replace(':', '-')}"))
    store_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(os.path.join(store_path, "models"))
    models_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path = Path(os.path.join(store_path, "tensorboard"))
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    for file in os.listdir(current_path):
        if file.endswith(".py"):
            print("copy: ", file)
            shutil.copyfile(os.path.join(current_path, file), os.path.join(store_path, file))
    print(f"python files are stored. Path: {store_path}")

    return store_path

def show_tensor_as_picture(img):
    img = img.cpu().detach()
    if len(img.shape)==4:
        img = img[0]
    if img.shape[0] == 3 or img.shape[0] == 1:
        img = img.permute(1,2,0)
    if img.shape[0] == 1:
        plt.imshow(img, cmap='gray', interpolation='nearest')
    else:
        plt.imshow(img, interpolation='nearest')
    plt.show()

def normalize_image(image):
    if torch.is_tensor(image):
        image_min = image.min()
        image_max = image.max()
        return (image-image.min())  / (image.max() - image.min())
    else:
        print("kann nicht normalisieren, da kein Tensor")
        exit(32)

def add_norm_noise(original, sigma, a=0, b=1, norm=True):
    """
    Args:
        original (torch.Tensor): Der Eingabetensor (b, c, w, h).
        sigma (float): DIE gewünschte Rausch-Stärke als std
        min_value: min Wert vom Datasatz
        max_value: max Wert vom Datasatz
        a: gewünschte Untergrenze vom Batch der Normalisierung
        b: gewünschte Obergrenze vom Batch der Normalisierung
        norm: True
    """
    noise = original + torch.randn_like(original) * sigma
    if norm:
        #noise = (noise-min_value) / (max_value-min_value) * (b-a)+a
        noise = ((noise-noise.min()) / (noise.max()-noise.min())) * (b-a)+a
    return noise

def add_noise_snr(x, snr_db):
    """
    Args:
        input_tensor (torch.Tensor): Der Eingabetensor (b, c, w, h).
        snr_db (float): Das gewünschte Signal-Rausch-Verhältnis in Dezibel (dB).
    Return:
        noise image: tensor of x + noise
        alpha: skaling faktor for noise - also interpreted as sigma
    """
    noise = torch.randn_like(x)
    snr_linear = 10 ** (snr_db / 10.0)
    snr_linear = snr_db
    Es = torch.sum(x**2)
    En = torch.sum(noise**2)
    alpha = torch.sqrt(Es/(snr_linear*En))
    noise = x + noise * alpha
    return noise, alpha.item()


def estimate_opt_sigma(noise_images, denoised, kmc, l_in, l_ex, n):
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
    for i in range(config.methods['n2info']['predictions']):# TODO: checken ob richhtiger wert aus k_mc
        # sample uniform between 0 and max(pixel count in images) exacly "samples" pixels
        indices = torch.randperm(all_pixels)[:m]
        # transform n into vector view and extract the sampled values
        sampled_values = n.view(-1)[indices]
        sorted_values = torch.sort(sampled_values).values #result from sort: [values, indices]

        
        e_l += torch.mean((n - sorted_values) ** 2)
    e_l = e_l / config.methods['n2info']['predictions']
    #Equation 6
    sigma = l_ex + (l_ex**2 + all_pixels*(l_in-e_l)).sqrt()/all_pixels#TODO: e_l ist sehr groß und l_in sehr klein -> NaN weegen wurzel
    return sigma

def estimate_opt_sigma_new(noise_images, dataLoader, model, device, sigma_info):
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
    n = torch.zeros(len(dataLoader)*batchsize, dimension)
    for batch_idx, (original, label) in enumerate((dataLoader)):
        if config.useSigma:
            noise_images = add_norm_noise(original, config.sigma, a=-1, b=1, norm=False)
            true_noise_sigma = config.sigma
        else:
            noise_images, true_noise_sigma = add_noise_snr(original, snr_db=config.sigmadb)
        noise_images = noise_images.to(device)
        _, denoised, loss_inv_tmp, loss_ex_tmp, marked_points = n2info(noise_images, model, device, sigma_info)
        loss_in += loss_inv_tmp
        loss_ex += loss_ex_tmp
        all_marked_points += marked_points
        #save every picture as vector in corresponding row of n
        n[batch_idx*batchsize : batch_idx*batchsize + batchsize, :] = denoised.view(batchsize, -1)
    loss_ex = loss_ex / all_marked_points
    loss_in = loss_in / len(dataLoader)
    for i in range(config.methods['n2info']['predictions']):
        # sample uniform between 0 and max(pixel count in all images) exacly "dimension" pixels
        indices = torch.randperm(all_pixels)[:dimension] #dimension = m in paper
        # transform n into vector view and extract the sampled values
        sampled_values = n.view(-1)[indices]
        sorted_values = torch.sort(sampled_values).values #result from sort: [values, indices]

        
        e_l += torch.mean((n - sorted_values) ** 2)
    e_l = e_l / config.methods['n2info']['predictions']
    #estimated_sigma= loss_ex**0.5 + (loss_ex + loss_in - e_l)**0.5 # version from github https://github.com/dominatorX/Noise2Info-code/blob/master/network_keras.py#L106
    estimated_sigma = loss_ex + (loss_ex**2 + dimension*(loss_in-e_l)).sqrt()/dimension #version from paper
    return estimated_sigma