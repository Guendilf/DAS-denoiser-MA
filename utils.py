from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil

import torch


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

def filp_lr_ud(img, lr, ud):
    """
    augmentation, flip leftt and right side (dim=2), fllip up and down side (dim=3)
    Args:
        img (tensor b,c,w,h): Image TTensor whichh could be flipt
        lr (int): flip left and right
        ud (int): flip up and down
    """
    if lr > 0:
        img = torch.flip(img, dims=[2])
    if ud > 0:
        img = torch.flip(img, dims=[3])
    return img

def estimate_opt_sigma(noise_images, denoised, samples, l_in, l_ex):
    """
    used for Noise2Info to determin a better suited sigma for loss calculation
    Args:
        moodel: the used torch model
        noise_images (tensor): images withh noise with sahpe: (b,c,w,h)
        samples: samples for Monte Carloo integration
    Returns:
        best sigma value
    """
    e_l = 0

    n = torch.sort(denoised-noise_images).values
    all_pixels = n.shape[0]*n.shape[1]*n.shape[2]*n.shape[3]
    for i in range(samples):
        # sample uniform between 0 and max(pixel count in images) exacly "samples" pixels
        indices = torch.randperm(all_pixels)[:samples]
        # transform n into vector view and extract the sampled values
        sampled_values = n.view(-1)[indices]
        sorted_values = torch.sort(sampled_values).values #result from sort: [values, indices]

        start_idx = i * samples
        end_idx = start_idx + samples
        for j in range(start_idx, end_idx):
            e_l += torch.sum((n.view(-1)[j] - sorted_values) ** 2)
    e_l = e_l / samples
    #Equation 6
    sigma = l_ex + (l_ex**2 + all_pixels*(l_in-e_l)).sqrt()/all_pixels#TODO: e_l ist sehr groß und l_in sehr klein -> NaN weegen wurzel
    return sigma