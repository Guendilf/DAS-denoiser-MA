from utils import *
from masks import Mask
from transformations import *
import torch

def n2noise(original, noise_images, sigma, device, model):
    #src1 = add_gaus_noise(original, 0.5, sigma).to(device)
    #schöner 1 Zeiler:
    #noise_image2 = add_norm_noise(original, sigma+0.3, min_value, max_value, a=-1, b=1)
    #there is no 2. input -> generate one
    if type(sigma) == int:
        noise_image2, alpha = add_noise_snr(noise_images, snr_db=sigma+2)
        noise_image2 = noise_image2.to(device) #+ mean
    #2. Input is conttained in sigma
    else:
        noise_image2 = sigma
    # Denoise image
    denoised = model(noise_images)
    loss_function = torch.nn.MSELoss()
    return loss_function(denoised, noise_image2), denoised


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


def n2void(original_images, noise_images, model, device, num_patches_per_img, windowsize, num_masked_pixels, augmentation):
    patches, clean_patches = generate_patches_from_list(noise_images, original_images, num_patches_per_img=num_patches_per_img, augment=augmentation)
    mask  = Mask.n2void_mask(patches, num_masked_pixels=8).to(device)
    #mask, num_masked_pixels_total = Mask.mask_random(patches, 8, mask_size=(1,1))
    #mask = mask.to(device)
    masked_noise = Mask.exchange_in_mask_with_pixel_in_window(mask, patches, windowsize, num_masked_pixels)
    
    denoised = model(masked_noise)
    denoised_pixel = denoised * mask
    target_pixel = patches * mask
    
    loss_function = torch.nn.MSELoss() #TODO: richtigge Loss, funktion?
    loss = torch.mean((denoised_pixel - target_pixel)**2)
    return loss_function(denoised_pixel, target_pixel), denoised, patches, clean_patches

def n2same(noise_images, device, model, lambda_inv=2):
    #mask, marked_points = Mask.mask_random(noise_images, maskamount=0.005, mask_size=(1,1))

    total_area = noise_images.shape[1]*noise_images.shape[2]*noise_images.shape[3]
    maskamount = int(np.round(0.005*total_area/1))
    mask=[]
    for _ in range(noise_images.shape[0]):
        #num_pixels = img.shape[1] * img.shape[2] *img.shape[3]
        # Erzeuge zufällige Indizes für die ausgewählten maskierten Pixel
        masked_indices = torch.randperm(total_area)[:maskamount]
        mask_img = torch.zeros(noise_images.shape[1], noise_images.shape[2], noise_images.shape[3])
        # Pixel in Maske auf 1 setzen
        mask_img.view(-1)[masked_indices] = 1
        # Mache für alle Chanels
        #mask = mask.unsqueeze(0).expand(1, -1, -1,-1)
        mask.append(mask_img)
    mask = torch.stack(mask)



    mask = mask.to(device)
    masked_input = (1-mask) * noise_images #delete masked pixels in noise_img
    masked_input = masked_input + (torch.normal(0, 0.2, size=noise_images.shape).to(device) * mask ) #deleted pixels will be gausian noise with sigma=0.2 as in appendix D
    denoised = model(noise_images)
    denoised_mask = model(masked_input)
    mse = torch.nn.MSELoss()
    loss_rec = torch.mean((denoised-noise_images)**2) # mse(denoised, noise_images)
    loss_inv = torch.sum(mask*(denoised-denoised_mask)**2)# mse(denoised, denoised_mask)
    loss = loss_rec + lambda_inv *1* (loss_inv/marked_points).sqrt()
    return loss, denoised, denoised_mask #J = count of maked_points

def self2self(noise_images, model, device, dropout_rate):
    
    #my methode idea:
    mask, marked_points = Mask.mask_random(noise_images, 0.5, (1,1))
    mask = mask.to(device)
    mask = (1-mask)
    """
    #original methode:  bernoulli mask with only (0 and 1-dropout)
    mask = torch.ones(noise_images.shape).to(device)
    mask = torch.nn.functional.dropout(mask, dropout_rate) * (1-dropout_rate)
    marked_points = torch.count_nonzero(1-mask)
    """
    #augmentation if value > 0
    flip_lr = np.random.randint(2)
    flip_ud = np.random.randint(2)
    
    masked_input = noise_images * mask
    masked_input = filp_lr_ud(masked_input, flip_lr, flip_ud)
    

    denoised, mod_mask = model(masked_input, mask)
    loss = torch.sum((denoised - noise_images)**2 * (1-mask)) / marked_points
    denoised = filp_lr_ud(denoised, flip_lr, flip_ud)
    return loss, denoised, mask, flip_lr, flip_ud

def noise2info(noise_images, model, device, sigma_start):
    #TODO: augmenttation
    #TODO: ich glaube mask ist genauso wie bei n2same. Weil: Paper S.6 unten
    #mask, marked_points = Mask.cut2self_mask((noise_images.shape[2],noise_images.shape[3]), noise_images.shape[0], mask_size=(1, 1), mask_percentage=0.005) #0,5% Piel maskieren
    mask, marked_points = Mask.mask_random(noise_images, 0.005, mask_size=(1,1))
    mask = mask.to(device)
    #mask = mask.unsqueeze(1)  # (b, 1, w, h)
    #mask = mask.expand(-1, noise_images.shape[1], -1, -1) # (b, 3, w, h)
    masked_input = (1-mask) * noise_images

    denoised = model(noise_images)
    denoised_mask = model(masked_input)
    loss_inv = torch.mean((denoised-noise_images)**2)
    loss_ex = torch.sum(mask*(denoised-denoised_mask)**2)
    loss = loss_inv + 2*sigma_start*(loss_ex/marked_points).sqrt() 
    return loss, denoised, loss_inv, loss_ex

def n2n_loss_for_das(denoised, target):
    loss_function = torch.nn.MSELoss()
    loss = loss_function(target, denoised)
    #len(noisy) = anzahl an Bildern
    return 1/len(target) *loss #+c #c = varianze of noise




def calculate_loss(model, device, dataLoader, methode, sigma, true_noise_sigma, batch_idx, original, noise_images, augmentation=True, 
                   lambda_inv=2, dropout_rate=0.3, samples=10, num_patches_per_img=None, num_masked_pixels=8, sigma_info=1):
    lr = 0
    ud = 0
    est_sigma_opt = -1
    if "n2noise" in methode:
        if "2_input" in methode:
            noise_image2, alpha = add_noise_snr(original, snr_db=sigma) #alpha = 0.25
            loss, denoised = n2noise(original, noise_images, noise_image2, device, model)
        else:
            loss, denoised = n2noise(original, noise_images, sigma, device, model)
    elif methode == "n2score":
        loss, denoised = n2score(noise_images, sigma_min=0.01, sigma_max=0.3, q=batch_idx/len(dataLoader), 
                                                        device=device, model=model, methode=methode)
        #tweedie - funktion for Gaus
        denoised = noise_images + true_noise_sigma**2*denoised

    elif "n2self" in methode:
        loss, denoised, masked_noise_image = n2self(noise_images, batch_idx, model)

    elif "n2void" in methode:
        #normalise Data as in github
        mean_noise = noise_images.mean(dim=[0,2,3])
        std_noise = noise_images.std(dim=[0,2,3])
        noise_images = (noise_images - mean_noise[None, :, None, None]) / std_noise[None, :, None, None]
        #mean_clean = original.mean(dim=[0,2,3])
        #std_clean = original.std(dim=[0,2,3])
        #original = (original - mean_clean[None, :, None, None]) / std_clean[None, :, None, None]

        loss, denoised, patches, original_patches = n2void(original, noise_images, model, device, num_patches_per_img=num_patches_per_img, windowsize=5, num_masked_pixels=num_masked_pixels, augmentation=augmentation)
        noise_images = patches
        original = original_patches
    elif "n2same"  in methode:
        #normalise Data as in github
        #mean_noise = noise_images.mean(dim=[0,2,3])
        #std_noise = noise_images.std(dim=[0,2,3])
        #noise_images = (noise_images - mean_noise[None, :, None, None]) / std_noise[None, :, None, None]

        loss, denoised, denoised_mask = n2same(noise_images, device, model, lambda_inv=lambda_inv)
    elif "self2self" in methode:
        loss, denoised, mask, lr, ud = self2self(noise_images, model, device, dropout_rate)

    elif "n2info" in methode:
        loss, denoised, loss_inv, loss_ex = noise2info(noise_images, model, device, sigma_info)
        #callculate new sigma at end of epoch
        """
        if batch_idx == len(dataLoader):
            with torch.no_grad():
                est_sigma_opt = estimate_opt_sigma(noise_images, denoised, samples, loss_inv, loss_ex).item()
            if est_sigma_opt < sigma_info:
                sigma_info = est_sigma_opt
                print("new sigma!")
            if sigma_info < 0:
                sigma_info = 0
                raise Exception(f"Optimal sigma in Noise2Info is negative with {sigma_info} . Loss_inv = {loss_inv} , Loss_ex = {loss_ex}!")
        """


    return loss, denoised, original, noise_images, (lr, ud, sigma_info, est_sigma_opt) #original, noise_images  are onlly if n2void