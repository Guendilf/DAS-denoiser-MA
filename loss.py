from utils import *
from masks import Mask
from transformations import *
import torch

def n2noise(original, noise_images, sigma, device, model, min_value, max_value, a=-1, b=1):
    #src1 = add_gaus_noise(original, 0.5, sigma).to(device)
    #schöner 1 Zeiler:
    #noise_image2 = add_norm_noise(original, sigma+0.3, min_value, max_value, a=-1, b=1)
    noise_image2, alpha = add_noise_snr(original, snr_db=sigma+2)
    noise_image2 = noise_image2.to(device) #+ mean
    # Denoise image
    denoised = model(noise_images)
    loss_function = torch.nn.MSELoss()
    return loss_function(denoised, noise_image2), denoised, noise_image2


def n2score(noise_images, sigma_min, sigma_max, q, device, model, methode): #q=batchindex/dataset
    u = torch.randn_like(noise_images).to(device)
    sigma_a = sigma_max*(1-q) + sigma_min*q
    vectorMap = model(noise_images+sigma_a*u)
    loss = torch.mean((sigma_a * vectorMap + u)**2)
    return loss, vectorMap


def n2self(noise_image, batch_idx, model):
    masked_noise_image, mask = Mask.n2self_mask(noise_image, batch_idx)
    denoised = model(masked_noise_image)
    #j_invariant_denoised = n2self_infer_full_image(noise_image, model)  #TODO: weiß noch nicht was das ist und wofür es benutzt wird
    return torch.nn.MSELoss()(denoised*mask, noise_image*mask), denoised, masked_noise_image


def n2void(original_images, noise_images, model, device, num_patches_per_img, windowsize, num_masked_pixels):
    patches, clean_patches = generate_patches_from_list(noise_images, original_images, num_patches_per_img=num_patches_per_img)
    mask  = Mask.n2void_mask(patches, num_masked_pixels=8).to(device)
    masked_noise = Mask.exchange_in_mask_with_pixel_in_window(mask, patches, windowsize, num_masked_pixels)
    
    denoised = model(masked_noise)
    denoised_pixel = denoised * mask
    target_pixel = patches * mask
    
    loss_function = torch.nn.MSELoss() #TODO: richtigge Loss, funktion?
    loss = torch.mean((denoised_pixel - target_pixel)**2)
    return loss_function(denoised_pixel, target_pixel), denoised, patches, clean_patches

def n2same(noise_images, device, model, lambda_inv=2):
    mask, marked_points = Mask.cut2self_mask((noise_images.shape[2],noise_images.shape[3]), noise_images.shape[0], mask_size=(1, 1), mask_percentage=0.005) #0,5% Piel maskieren
    mask = mask.to(device)
    mask = mask.unsqueeze(1)  # (b, 1, w, h)
    mask = mask.expand(-1, noise_images.shape[1], -1, -1) # (b, 3, w, h)
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
        loss, denoised, noise_image2 = n2noise(original, noise_images, sigma, device, model, min_value2, max_value2)
    elif methode == "n2score":
        loss, denoised = n2score(noise_images, sigma_min=0.01, sigma_max=0.3, q=batch_idx/len(dataLoader), 
                                                        device=device, model=model, methode=methode)
        #tweedie - funktion for Gaus
        denoised = noise_images + sigma**2*denoised

    elif "n2self" in methode:
        loss, denoised, masked_noise_image = n2self(noise_images, batch_idx, model)

    elif methode == "n2void":
        #normalise Data as in github
        mean_noise = noise_images.mean(dim=[0,2,3])
        std_noise = noise_images.std(dim=[0,2,3])
        noise_images = (noise_images - mean_noise[None, :, None, None]) / std_noise[None, :, None, None]
        mean_clean = original.mean(dim=[0,2,3])
        std_clean = original.std(dim=[0,2,3])
        original = (original - mean_clean[None, :, None, None]) / std_clean[None, :, None, None]

        loss, denoised, patches, original_patches = n2void(original, noise_images, model, device, num_patches_per_img=None, windowsize=5, num_masked_pixels=8)
        noise_images = patches
        original = original_patches
    elif "n2same"  in methode:
        #normalise Data as in github
        mean_noise = noise_images.mean(dim=[0,2,3])
        std_noise = noise_images.std(dim=[0,2,3])
        noise_images = (noise_images - mean_noise[None, :, None, None]) / std_noise[None, :, None, None]

        loss, denoised, denoised_mask = n2same(noise_images, device, model, lambda_inv=2)
    return loss, denoised, original, noise_images #original, noise_images  are onlly if n2void