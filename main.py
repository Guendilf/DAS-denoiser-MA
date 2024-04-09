import math
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

from N2N_Unet import N2N_Unet_DAS, N2N_Unet_Claba, Cut2Self

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


def cut2self_masl(image_size, batch_size, mask_size=(4, 4), mask_percentage=0.003):
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


"""
TRAIN FUNKTIONS
"""



def train(model, optimizer, device, dataLoader, dataset):
    #mit lossfunktion aus "N2N(noiseToNoise)"
    #kabel ist gesplitet und ein Teil (src) wird im Model behandelt und der andere (target) soll verglichen werden
    #writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))
    #writer = SummaryWriter(log_dir="log/tensorboard")
    loss_log = []
    psnr_log = []
    for batch_idx, (src, target) in enumerate(tqdm(dataLoader)): #src.shape=[batchsize, rgb, w, h]
        if batch_idx == max_Iteration:
            break
        src1 = add_gaus_noise(src, 0.5, 0.1**0.5).to(device)
        src1 = src + torch.randn_like(src) * sigma + mean
        target = src + torch.randn_like(src) * sigma + mean
        target = add_gaus_noise(src, 0.6, 0.4**0.5).to(device)
       
        # Denoise image
        denoised = model(src1)

        #calculate loss
        loss = orig_n2n_loss(denoised, target)
        loss_log.append(loss.item())
        #writer.add_scalar("Loss", loss, batch_idx)
        #writer.flush()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        psnr_batch = calculate_psnr(target, denoised)
        psnr_log.append(psnr_batch.item())
        if batch_idx % 100 == 0:
            show_pictures_from_dataset(src, model, batch_idx)
    #writer.close()
    plt.show()
    return loss_log, psnr_log

        

def main(argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    celeba_dir = 'dataset/celeba_dataset'
    dataset = datasets.CelebA(root=celeba_dir, split='train', download=True, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)

    mask = cut2self_masl((128,128), 64).to(device)

    #model = N2N_Unet_Claba().to(device)
    model = Cut2Self(mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    
    print(f"Using {device} device")

    loss, psnr_log = train(model, optimizer, device, dataLoader, dataset)
    plt.plot(loss, color='blue')
    plt.plot(psnr_log, color='red')
    plt.show()
    print(loss)
    print(psnr_log)
    #show_pictures_from_dataset(dataset, model)
    
    print("fertig\n")
    


if __name__ == "__main__":
    app.run(main)