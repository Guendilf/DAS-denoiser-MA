import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import *
from tqdm import tqdm
from metric import Metric
import bm3d

sigma = 2

transform_noise = transforms.Compose([
    transforms.CenterCrop((128,128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float()),
    transforms.Lambda(lambda x:  x * 2 -1),
    ])
celeba_dir = 'dataset/celeba_dataset'   
dataset = datasets.CelebA(root=celeba_dir, split='train', download=False, transform=transform_noise)
dataset = torch.utils.data.Subset(dataset, list(range(6400)))
dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)



dataset_validate = datasets.CelebA(root=celeba_dir, split='valid', download=False, transform=transform_noise)
dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(640)))

dataset_test = datasets.CelebA(root=celeba_dir, split='test', download=False, transform=transform_noise)
dataset_test = torch.utils.data.Subset(dataset_test, list(range(640)))

dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=False)
dataLoader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

psnr_bm3d = []
sim_bm3d = []
for batch_idx, (original, label) in enumerate(tqdm(dataLoader)):
    noise_images, true_noise_sigma = add_noise_snr(original, snr_db=sigma)
    noise_images = torch.permute(noise_images, (0,2,3,1)).numpy()
    #original = torch.permute(original, (0,2,3,1)).numpy()
    #generatte BM3D Results for every image in batch
    denoised_bm3d_batch = []
    for img in noise_images:
        denoised = bm3d.bm3d(img, true_noise_sigma)
        denoised_bm3d_batch.append(denoised)
    denoised_bm3d_batch = np.stack(denoised_bm3d_batch, axis=0)
    denoised_bm3d_batch = torch.permute(torch.from_numpy(denoised_bm3d_batch), (0,3,1,2))
    #calculatte PSNR and Similarity
    denoised_bm3d_batch = (denoised_bm3d_batch-denoised_bm3d_batch.min())  / (denoised_bm3d_batch.max() - denoised_bm3d_batch.min())
    original = (original-original.min())  / (original.max() - original.min())
    psnr_batch = Metric.calculate_psnr(original, denoised_bm3d_batch)
    similarity_batch, diff_picture = Metric.calculate_similarity(original, denoised_bm3d_batch)
    psnr_bm3d.append(psnr_batch.item())
    sim_bm3d.append(similarity_batch)

print("PSNR AVG in Batches:")
print(psnr_bm3d)
print()
print("SIM AVG in Batches:")
print(sim_bm3d)
print()
print(f" PSNR AVG: {Metric.avg_list(psnr_bm3d)}")
print(f" SIM AVG: {Metric.avg_list(sim_bm3d)}")
