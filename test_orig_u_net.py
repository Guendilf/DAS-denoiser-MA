import math
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

from N2N_Unet import U_Net_origi

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#sys.path.insert(0, str(Path(__file__).resolve().parents[3])) #damit die Pfade auf dem Server richtig waren (copy past von PG)
from absl import app
from torch.utils.tensorboard import SummaryWriter


def main(argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    #celeba_dir = 'dataset/celeba_dataset'
    #dataset = datasets.CelebA(root=celeba_dir, split='train', download=True, transform=transform)
    #dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)

    

    #model = N2N_Orig_Unet().to(device)
    model = U_Net_origi().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    batch = torch.randn((64,3,572,572)).to(device)
    print("Input shape: " + str(batch.shape) + "\n")
    denoised = model(batch)
    print("Input shape: " + str(denoised.shape) + "\n")
    print("soll: (64 x 3 x 388 x 388)")

    #calculate loss
    loss_function = torch.nn.MSELoss()
    loss = loss_function(batch, denoised)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("fertig")





if __name__ == "__main__":
    app.run(main)