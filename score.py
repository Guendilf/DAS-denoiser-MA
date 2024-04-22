import sys
import time
from pathlib import Path
from tqdm import tqdm

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

def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)
    max_intensity = 1.0  # weil Vild [0, 1]

    psnr = 10 * torch.log10((max_intensity ** 2) / mse) #mit epsilon weil mse gegen 0 gehen kann
    return psnr


#Stein's Unbiased Eisk Estimate     Kapitel 2.3
def loss_SURE (output, target, model, sigma):
    mse = torch.nn.MSELoss()(output, target)

    #TODO: Monte-Carlo-Simulation
    divergences = []
    for _ in range(output.shape[0]):
        # random noise pictures
        new_noise = torch.randn_like(output) * sigma

        # Berechnen Sie den Gradienten von f_lambda(y) bezüglich y
        #new_noise.requires_grad = True
        output = model(output + new_noise)
        gradient = torch.autograd.grad(outputs=output, inputs=new_noise, grad_outputs=torch.ones_like(output))

        # rechne Divergenz von f_lambda(y)
        divergence = torch.sum(gradient[0])
        #divergence = torch.sum(new_noise.grad)
        divergences.append(divergence.item())
    divergence = torch.mean(torch.tensor(divergences))
    

    regularisation = 2 * sigma**2 * divergence * output #TODO: wie berechne ich die divergenz?
    return torch.mean(mse + regularisation), output


def tweedie(pictures):
    pass
    return denoised


def noise_mode (mode, y, loss, sigma=0, zeta=0, alpha=0, beta=0):
    if mode == "gaus":
        return y + sigma**2 * loss
    elif mode == "poisson":
        return (y + zeta/2) * np.exp(loss)
    elif mode == "gamma":
        return (beta * y) / ((alpha -1) - y*loss)
    else:
        print("ERROR! for noise mode")
        return None



def train(model, optimizer, device, dataLoader, dataset, sigma):

    loss_log = []
    psnr_log = []
    for batch_idx, (original, label) in enumerate(tqdm(dataLoader)): #src.shape=[batchsize, rgb, w, h]
        if batch_idx == max_Iteration:
            break
        
        u = torch.randn_like(original).to(device)
        sigma_a = (torch.randn_like(original) * sigma).to(device)
        picture = (original + sigma_a*u).to(device)
        #target = (original + torch.randn_like(original) * sigma + 0.5).to(device)

        #calculate loss
        #loss, src1 = loss_SURE(src1, target, model, sigma)
        denoised = model(picture)
        loss = ((original - denoised)**2).sum().sqrt()
        loss = loss / original.numel()
        loss_log.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        psnr_batch = calculate_psnr(original, denoised)
        psnr_log.append(psnr_batch.item())

    return loss_log, psnr_log

        

def main(argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    celeba_dir = 'dataset/celeba_dataset'
    dataset = datasets.CelebA(root=celeba_dir, split='train', download=True, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)


    model = N2N_Orig_Unet(input_chanels=3, output_chanels=3).to(device)
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