import os
import time
from tqdm import tqdm
from skimage.metrics import structural_similarity as sim

import config

from models.N2N_Unet import N2N_Unet_DAS, N2N_Orig_Unet, Cut2Self, U_Net_origi, U_Net, TestNet
from metric import Metric
from masks import Mask
from utils import *
from transformations import *

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from absl import app
from torch.utils.tensorboard import SummaryWriter

num_mc = 100
sigma_n = 1
max_epochs = 20

def get_lr_lambda(initial_lr, step_size, lr_decrement):
    def lr_lambda(step):
        #return max(0.0, initial_lr - (step // step_size) * lr_decrement / initial_lr)
        exp = step / step_size
        exp = int(exp) #staircase
        return lr_decrement ** exp #*initial_lr TODO: muss ich nicht nochh die multiplikation machhen
    return lr_lambda


def augment_patch(patch):
    patch = np.rot90(patch, k=np.random.randint(4), axes=(1, 2))
    patch = np.flip(patch, axis=-2) if np.random.randint(2) else patch
    return patch

def get_stratified_coords2D(coord_gen, box_size, shape):
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    x_coords = []
    y_coords = []
    for i in range(box_count_y):
        for j in range(box_count_x):
            y, x = coord_gen
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                y_coords.append(y)
                x_coords.append(x)
    return (y_coords, x_coords)

def rand_float_coords2D(boxsize):
    return (np.random.rand() * boxsize, np.random.rand() * boxsize)

def input_fn(sources, patch_size, mask_perc):

    def get_patch(source):
        valid_shape = source.shape[1:3] - np.array(patch_size)  # Assume source shape is (C, H, W)
        if any([s <= 0 for s in valid_shape]): #True if patch_size == Picture_size
            source_patch = augment_patch(source)
        else:
            #coords = [np.random.randint(0, shape_i + 1) for shape_i in valid_shape]
            x = np.random.randint(0, valid_shape[0] + 1)
            y = np.random.randint(0, valid_shape[1] + 1)
            coords = [x,y]
            s = tuple([slice(0, source.shape[0])] + [slice(coord, coord + size) for coord, size in zip(coords, patch_size)])
            source_patch = augment_patch(source[s])

        mask = np.zeros_like(source_patch)
        for c in range(source.shape[0]):
            boxsize = np.round(np.sqrt(100 / mask_perc))
            maskcoords = get_stratified_coords2D(rand_float_coords2D(boxsize), box_size=boxsize, shape=patch_size)
            indexing = (c,) + maskcoords
            mask[indexing] = 1.0
        return source_patch, np.random.normal(0, 0.2, source_patch.shape), mask

    patches = [get_patch(source) for source in sources] #list from tensor
    source_patches, nooise_input, mask = zip(*patches)
    source_patches = [torch.from_numpy(arr.copy()) for arr in source_patches]
    source_patches = torch.stack(source_patches)
    nooise_input = [torch.from_numpy(arr) for arr in nooise_input]
    nooise_input = torch.stack(nooise_input)
    mask = [torch.from_numpy(arr) for arr in mask]
    mask = torch.stack(mask)

    return source_patches, nooise_input, mask



def n2same(noise_images, device, model, mask, lambda_inv=2):
    if mask is None:
        mask, marked_points = Mask.mask_random(noise_images, maskamount=0.005, mask_size=(1,1))
        mask = mask.to(device)
        masked_input = (1-mask) * noise_images #delete masked pixels in noise_img
        masked_input = masked_input + (torch.normal(0, 0.2, size=noise_images.shape).to(device) * mask ) #deleted pixels will be gausian noise with sigma=0.2 as in appendix D
    else:
        marked_points = torch.sum(mask)
        masked_input = (1-mask) * noise_images + (torch.normal(0, 0.2, size=noise_images.shape).to(device) * mask)

    denoised = model(noise_images)
    denoised_mask = model(masked_input)
    mse = torch.nn.MSELoss()
    loss_rec = torch.mean((denoised-noise_images)**2) # mse(denoised, noise_images)
    loss_inv = torch.sum(mask*(denoised-denoised_mask)**2)# mse(denoised, denoised_mask)
    loss = loss_rec + lambda_inv *sigma_n* (loss_inv/marked_points).sqrt()
    return loss, denoised, loss_rec, loss_inv, marked_points


def train(model, device, optimizer, scheduler, dataLoader, mode, writer, rauschen, lambda_inv, epoch, methode):
    global sigma_n
    losses = []
    psnr = []
    lex = 0
    lin = 0
    all_marked = 0
    n = torch.empty((0, 3*128*128)).to(device)
    for batch_idx, (original, label) in enumerate((dataLoader)):
        if "new masking" in rauschen:
            original, _, mask = input_fn(original, (128,128), 0.5)
            mask = mask.to(device)
        else:
            mask = None
        if "komplex" in rauschen:
            original = original*255
            poison = 255.0 * np.random.poisson(30.0*(original/255.0))/30.0
            gauss =  np.random.normal(0, 60, original.shape)
            bernoulli_noise_map = np.random.binomial(1, 0.5, original.shape)*255
            bernoulli_noised = np.where(np.random.uniform(0, 1, original.shape)<0.2, bernoulli_noise_map, gauss+poison)
            noise = np.clip(bernoulli_noised, 0, 255.0)
            noise_image = (torch.from_numpy(noise)/255.0).to(device).type(torch.float32)
            original = original/255
        else:
            noise = (torch.rand_like(original*255.0)*0.4).type(torch.float32)
            noise_image = torch.clip(original*255.0 + noise, 0,255.0)/255.0
        original = original.to(device)
        noise_image = noise_image.to(device)
        if "norm" in rauschen:
            noise_image = (noise_image-noise_image.mean() / noise_image.std())
            original = (original-original.mean() / original.std())
        if "train" in mode:
            loss, denoised, _, _, _ = n2same(noise_image, device, model, mask, lambda_inv)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            writer.add_scalar('Train loss', loss, epoch * len(dataLoader) + batch_idx)
            writer.add_scalar('Train psnr', Metric.calculate_psnr(original, denoised).item(), epoch * len(dataLoader) + batch_idx)
        elif "val" in mode:
            with torch.no_grad():
                loss, denoised, loss_rec, loss_inv, marked_pixel = n2same(noise_image, device, model, mask, lambda_inv)
                if "n2info" in methode:
                    all_marked += marked_pixel
                    lex += loss_rec
                    lin += loss_inv
                    n_partition = (denoised-noise_image).view(denoised.shape[0], -1) # (b, c*w*h)
                    n_partition = torch.sort(n_partition, dim=1).values #descending=False
                    n = torch.cat((n, n_partition), dim=0)
                    if batch_idx == len(dataLoader)-1:
                        e_l = 0
                        for i in range(num_mc): #kmc
                            #to big for torch.multinomial if all pictures from validation should be used
                            #samples = torch.tensor(torch.multinomial(n.view(-1), n.shape[1], replacement=True))#.view(1, n.shape[1])
                            #samples = torch.sort(samples).values
                            samples = np.sort(np.random.choice((n.cpu()).reshape(-1),[1, n.shape[1]])) #(1,49152)
                            e_l += torch.mean((n-torch.from_numpy(samples).to(device))**2)
                        lex = lex / (len(dataLoader) * denoised.shape[0])
                        lin = lin / all_marked
                        e_l = e_l / num_mc
                        #estimated_sigma = (lin)**0.5 + (lin + lex-e_l)**0.5 #inplementation from original github of noise2info
                        m = len(dataLoader) * denoised.shape[0] *3*128*128
                        estimated_sigma = lex + (lex**2 * m (lin-e_l))**0.5/m
                        print('new sigma_loss is ', estimated_sigma)
                        if 0 < estimated_sigma < sigma_n:
                            sigma_n = float(estimated_sigma)
                            print('sigma_loss updated to ', estimated_sigma)
                        writer.add_scalar('estimated sigma', estimated_sigma, epoch)
                        writer.add_scalar('lex', lex, epoch)
                        writer.add_scalar('lin', lin, epoch)
                        writer.add_scalar('e_l', e_l, epoch)
            writer.add_scalar('Val loss', loss, epoch * len(dataLoader) + batch_idx)
            writer.add_scalar('Val psnr', Metric.calculate_psnr(original, denoised).item(), epoch * len(dataLoader) + batch_idx)
        else:
            with torch.no_grad():
                denoised = model(noise_image)
            writer.add_scalar('Test psnr', Metric.calculate_psnr(original, denoised).item(), epoch * len(dataLoader) + batch_idx)
            loss = 0
        denoised = denoised * img_std + img_mean
        noise_image = noise_image * img_std + img_mean
        psnr.append(Metric.calculate_psnr(original, denoised))
        losses.append(loss)

    return loss, psnr
        


          

def main(argv):
    global sigma_n
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    device = "cpu"


    celeba_dir = config.celeba_dir
    methodeListe = ['n2info']#, 'n2same']
    #options = ['old', 'old new masking', 'old norm', 'old norm new masking', 'komplex', 'komplex new masking', 'komplex norm', 'komplex norm new masking']
    options = ['old', 'old new masking']
    #end_results = pd.DataFrame(columns=methodeListe)

    
    transform_noise = transforms.Compose([
        transforms.CenterCrop((128,128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        #transforms.Lambda(lambda x:  x * 2 -1),
        ])
    print("lade Datensätze ...")
    dataset = datasets.CelebA(root=celeba_dir, split='train', download=False, transform=transform_noise)
    dataset = torch.utils.data.Subset(dataset, list(range(6400)))
    #dataset = torch.utils.data.Subset(dataset, list(range(32)))
    
    dataset_validate = datasets.CelebA(root=celeba_dir, split='valid', download=False, transform=transform_noise)
    dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(640)))
    #dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(32)))

    dataset_test = datasets.CelebA(root=celeba_dir, split='test', download=False, transform=transform_noise)
    dataset_test = torch.utils.data.Subset(dataset_test, list(range(640)))
    #dataset_test = torch.utils.data.Subset(dataset_test, list(range(32)))

    store_path = log_files()
    for methode in methodeListe:
        #if "n2info" in methode:
            #dataset = torch.utils.data.Subset(dataset, list(range(960)))#960=30*battch_size
            #dataset_validate = torch.utils.data.Subset(dataset_validate, list(range(96)))
        for option in options:
            dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
            dataLoader_validate = DataLoader(dataset_validate, batch_size=32, shuffle=False)
            dataLoader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
            if torch.cuda.device_count() == 1:
                model = N2N_Orig_Unet(3,3).to(device) #default
            else:
                model = U_Net(first_out_chanel=96, batchNorm=True).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config.methodes[methode]['lr'])
            lr_lambda = get_lr_lambda(config.methodes[methode]['lr'], config.methodes[methode]['changeLR_steps'], config.methodes[methode]['changeLR_rate'])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            writer = SummaryWriter(log_dir=os.path.join(store_path, str(methode + ' ' + option) ,"tensorboard"))

            for epoch in tqdm(range(max_epochs)):
                loss, psnr = train(model, device, optimizer, scheduler, dataLoader, 'train', writer, rauschen=option, lambda_inv=2, epoch=epoch, methode=methode)

                loss, psnr = train(model, device, optimizer, scheduler, dataLoader_validate, 'val', writer, rauschen=option, lambda_inv=2, epoch=epoch, methode=methode)
            
            loss, psnr = train(model, device, optimizer, scheduler, dataLoader_test, 'test', writer, rauschen=option, lambda_inv=2, epoch=0, methode=methode)

            sigma_n = 1
    print(str(datetime.now().replace(microsecond=0)).replace(':', '-'))




if __name__ == "__main__":
    app.run(main)