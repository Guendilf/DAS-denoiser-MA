import math
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

from N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#sys.path.insert(0, str(Path(__file__).resolve().parents[3])) #damit die Pfade auf dem Server richtig waren (copy past von PG)
from absl import app
from torch.utils.tensorboard import SummaryWriter


max_Iteration = 100


transform = transforms.Compose([
    transforms.Resize((128, 128)), #TODO:crop und dann resize
    transforms.ToTensor(),                  # PIL-Bild in Tensor
    transforms.Lambda(lambda x: x.float()),  # in Float
    transforms.Lambda(lambda x: x / torch.max(x)) #skallieren auf [0,1]
])


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





"""
Pre-Processing FUNKTIONS
"""


def add_gaus_noise(image, mean, sigma):
    noise = torch.randn_like(image) * sigma + mean
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image


def cut2self_mask(image_size, batch_size, mask_size=(4, 4), mask_percentage=0.003):
    total_area = image_size[0] * image_size[1]
    
    # Berechnen Fläche jedes Quadrats in der Maske
    mask_area = mask_size[0] * mask_size[1]
    
    # Berechnen Anzahl der Quadratregionen
    num_regions = int((mask_percentage * total_area) / mask_area)
    # binäre Maske
    masks = []
    for _ in range(batch_size):
        mask = torch.zeros(image_size[0], image_size[1], dtype=torch.float32)
        for _ in range(num_regions):        # generiere eine maske
            x = torch.randint(0, image_size[0] - mask_size[0] + 1, (1,))
            y = torch.randint(0, image_size[1] - mask_size[1] + 1, (1,))
            mask[x:x+mask_size[0], y:y+mask_size[1]] = 1
        masks.append(mask)

    return torch.stack(masks, dim=0)



def n2self_mask( noise_image, i, grid_size=3):
    phasex = i % grid_size
    phasey = (i // grid_size) % grid_size
    mask = n2self_pixel_grid_mask(noise_image[0, 0].shape, grid_size, phasex, phasey)
    mask = mask.to(noise_image.device)
    mask_inv = torch.ones(mask.shape).to(noise_image.device) - mask
    if True: #mode == 'interpolate':
        masked = n2self_interpolate_mask(noise_image, mask, mask_inv)
    #elif self.mode == 'zero':
        #masked = X * mask_inv
    #else:
        #raise NotImplementedError
    #if self.include_mask_as_input:
        #net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
    #else:
    net_input = masked
    return net_input, mask

def n2self_pixel_grid_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                A[i, j] = 1
    return torch.Tensor(A)

def n2self_interpolate_mask(tensor, mask, mask_inv):
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = np.repeat(kernel[np.newaxis, np.newaxis, :, :], repeats=3, axis=1) #repeat = 3 weil 3 Chanel im Bild
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()
    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)
    return filtered_tensor * mask + tensor * mask_inv

#TODO: wann benutzen?
def n2self_infer_full_image(noise_image, model, n_masks=3): #n_masks=grid_size
    net_input, mask = n2self_mask(noise_image, 0)
    net_output = model(net_input)
    acc_tensor = torch.zeros(net_output.shape).cpu()
    for i in range(n_masks):
        net_input, mask = n2self_mask(noise_image, i)
        net_output = model(net_input)
        acc_tensor = acc_tensor + (net_output * mask).cpu()
    return acc_tensor
    



"""
LOSS FUNKTIONS  +  EVALUATION FUNKTIONS
"""


#original Loss funktion from N2N paper
def orig_n2n_loss(denoised, target):
    loss_function = torch.nn.MSELoss()
    return loss_function(denoised, target)


def n2n_loss_for_das(denoised, target):
    loss_function = torch.nn.MSELoss()
    loss = loss_function(target, denoised)
    #len(noisy) = anzahl an Bildern
    return 1/len(target) *loss #+c #c = varianze of noise


def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)
    max_intensity = 1.0  # weil Vild [0, 1]

    psnr = 10 * torch.log10((max_intensity ** 2) / mse) #mit epsilon weil mse gegen 0 gehen kann
    return psnr

def calculate_similarity(image1, image2):
    image_range = (image1.max() - image1.min()).item()
    im1 = image1.cpu().detach().numpy()
    im2 = image2.cpu().detach().numpy()
    similarity_index, diff_image = sim(im1, im2, data_range=image_range, channel_axis=1, full=True)
    return similarity_index, diff_image

"""
TRAIN FUNKTIONS
"""



def train(model, optimizer, device, dataLoader, dataset_test, methode, sigma, mode):
    #mit lossfunktion aus "N2N(noiseToNoise)"
    #kabel ist gesplitet und ein Teil (src) wird im Model behandelt und der andere (target) soll verglichen werden
    #writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))
    #writer = SummaryWriter(log_dir="log/tensorboard")
    loss_log = []
    psnr_log = []
    sim_log = []
    value_loss_log = []
    value_psnr_log = []
    value_sim_log = []
    if mode=="test":
        model.eval()
    else:
        model.train()
    for batch_idx, (original, label) in enumerate(tqdm(dataLoader)): #src.shape=[batchsize, rgb, w, h]
        if batch_idx == max_Iteration:
            break
        torch.set_grad_enabled(mode=="train")

        if methode == "n2n_orig":
            #src1 = add_gaus_noise(original, 0.5, sigma).to(device)
            #schöner 1 Zeiler:
            noise = torch.randn_like(original) * sigma
            src1 = (original + noise).to(device)
            noise_image = (original + torch.randn_like(original) * (sigma+0.3)).to(device) #+ mean
            # Denoise image
            denoised = model(src1)
            loss = orig_n2n_loss(denoised, noise_image)
            
        elif "score" in methode:
            u = torch.randn_like(original).to(device)
            sigma_a = (torch.randn_like(original) * sigma).to(device)
            noise = sigma_a*u
            noise_image = (original + noise).to(device)
            #loss, src1 = loss_SURE(src1, target, model, sigma)
            denoised = model(noise_image)
            if methode == "score_ar":
                loss = ((u + sigma_a*noise_image)**2).sum().sqrt()
            else:
                loss = ((original - denoised)**2).sum().sqrt()
            loss = loss / original.numel()


        elif methode == "n2self":
            noise = torch.randn_like(original) * sigma
            noise_image, mask = n2self_mask(original, batch_idx)
            denoised = model(noise_image)
            #j_invariant_denoised = n2self_infer_full_image(noise_image, model)  #TODO: weiß noch nicht was das ist und wofür es benutzt wird
            loss = torch.nn.MSELoss()(denoised*mask, noise_image*mask)
            

       


        
        loss_log.append(loss.item())
        #writer.add_scalar("Loss", loss, batch_idx)
        #writer.flush()

        if mode=="train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        psnr_batch = calculate_psnr(noise_image, denoised)
        similarity_batch, diff_picture = calculate_similarity(noise_image, denoised)
        psnr_log.append(psnr_batch.item())
        sim_log.append(similarity_batch)
        if mode == "train" and batch_idx % 10 == 0:
            #show_pictures_from_dataset(src, model, batch_idx)
            model.eval()
            dataLoader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)
            value_loss, value_psnr, value_sim,_,_,_ = train(model, optimizer, device, dataLoader_test, dataset_test, methode, sigma, mode="test")
            value_loss_log.append(sum(value_loss)/len(value_loss))
            value_psnr_log.append(sum(value_psnr)/len(value_psnr))
            value_sim_log.append(sum(value_sim)/len(value_sim))
    #writer.close()
    plt.show()
    return loss_log, psnr_log, sim_log, value_loss_log, value_psnr_log, value_sim_log

        

def main(argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methode="n2n_orig"

    celeba_dir = 'dataset/celeba_dataset'
    dataset = datasets.CelebA(root=celeba_dir, split='train', download=True, transform=transform)
    dataset_test = datasets.CelebA(root=celeba_dir, split='test', download=True, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)

    mask = cut2self_mask((128,128), 64).to(device)

    model = N2N_Orig_Unet(3,3).to(device)
    #model = Cut2Self(mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    
    print(f"Using {device} device")

    loss, psnr, similarity, value_loss, value_psnr, value_similarity = train(model, optimizer, device, dataLoader, dataset_test, methode, sigma=0.4, mode="train")
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



    print(loss)
    print(psnr)
    print(value_similarity)
    show_pictures_from_dataset(dataset, model)
    
    print("fertig\n")
    


if __name__ == "__main__":
    app.run(main)