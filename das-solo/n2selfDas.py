import numpy as np
import statistics
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split
from absl import app
from torch.utils.tensorboard import SummaryWriter

from import_files import log_files
from import_files import U_Net
from import_files import SyntheticNoiseDAS


epochs = 100 #2.000 epochen - 1 Epoche = 3424 samples
batchsize = 32
dasChanelsTrain = 11
dasChanelsVal = 11
dasChanelsTest = 11
lr = 0.0001
batchnorm = False
save_model = False

snr_level = log_SNR=(-2,4)#default, ist weas anderes
gauge_length = 30#real sind 19.2
snr = (np.log(0.01), np.log(10))
slowness = (0.2*10**3, 10*10**3) #angabe in m/s, laut paper 0.2 bis 10 km/s

modi = 0


"""
TODO:
- swuish activation funktion with ??? parameter
- gauge = 19,2
- 50 Hz frequenz
- anderer loss?
"""
def show_das(original):
    if original is torch.Tensor:
        original = original.to('cpu').detach().numpy()
    original = original[0]
    plt.figure(figsize=(7, 11))
    for i in range(original.shape[1]):
        #sr = original[0][i] / original[0][i].std()
        sr = original[0][i]
        plt.plot(sr + 3*i, c="k", lw=0.5, alpha=1)
        #if every chanle by it self
        #plt.subplot(original.shape[1], 1, i + 1)
        #plt.plot(original[0, i].numpy(), c="k", lw=0.5, alpha=1)
        #plt.title(f'Channel {i + 1}')
        plt.tight_layout()
    plt.show()
    
def saveAndPicture(psnr, clean, noise_images, denoised, mode, writer, epoch, len_dataloader, batch_idx, model, store):
    comparison = torch.cat((clean[:1], denoised[:1], noise_images[:1]), dim=0)
    comparison = comparison[:,:,:,:512]
    grid = make_grid(comparison, nrow=1, normalize=False).cpu()
    if mode == "train":
        writer.add_image('Denoised Training', grid, global_step=epoch * len_dataloader + batch_idx)
    elif mode == "validate":
        writer.add_image('Denoised Validation', grid, global_step=epoch * len_dataloader + batch_idx)
    else:
        writer.add_image('Denoised Test', grid, global_step=1 * len_dataloader + batch_idx)
    if "test" not in mode:
        print(f"best model found with psnr: {psnr}")
        model_save_path = os.path.join(store, "models", f"{round(psnr, 1)}-{mode}-{epoch}-{batch_idx}.pth")
        if save_model:
            torch.save(model.state_dict(), model_save_path)
        else:
            f = open(model_save_path, "x")
            f.close()

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
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel)
    kernel = kernel / kernel.sum()
    kernel = np.repeat(kernel, repeats=tensor.shape[1], axis=1).to(device)
    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)
    return filtered_tensor * mask + tensor * mask_inv #TODO:verschieben in mask
def mask(noise_image, i, grid_size=3, mode='interpolate', include_mask_as_input=False): 
    phasex = i % grid_size
    phasey = (i // grid_size) % grid_size
    mask = n2self_pixel_grid_mask(noise_image[0, 0].shape, grid_size, phasex, phasey)
    mask = mask.to(noise_image.device)
    mask_inv = 1 - mask
    mask_inv = torch.ones(mask.shape).to(noise_image.device) - mask
    if mode == 'interpolate':
        masked = n2self_interpolate_mask(noise_image, mask, mask_inv)
    elif mode == 'zero':
        masked = noise_image * mask_inv
    #else:
        #raise NotImplementedError
    if include_mask_as_input:
        net_input = torch.cat((masked, mask.repeat(noise_image.shape[0], 1, 1, 1)), dim=1)
    else:
        net_input = masked
    return net_input, mask

def calculate_loss(noise_image, model, batch_idx):
    #masked_noise_image, mask = Mask.n2self_mask(noise_image, batch_idx)
    mask = torch.zeros_like(noise_image).to(noise_image.device)
    for i in range(mask.shape[0]):
        mask[i, :, np.random.randint(0, mask.shape[2]), :] = 1
    masked_noise_image = (1-mask) * noise_image
    denoised = model(masked_noise_image)
    return torch.nn.MSELoss()(denoised*(mask), noise_image*(mask)), denoised

def train(model, device, dataLoader, optimizer, mode, writer, epoch, store_path, bestPsnr):
    global modi
    loss_log = []
    psnr_log = []
    scaledVariance_log = []
    for batch_idx, (noise_images, clean, noise, std, amp) in enumerate(dataLoader):
        clean = clean.to(device).type(torch.float32)
        noise_images = noise_images.to(device).type(torch.float32)
        std = std.to(device).type(torch.float32)
        if modi == 1:
            noise_images = noise_images*std
        if mode == "train":
            model.train()
            loss, denoised = calculate_loss(noise_images, model, batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            """
            if config.methodes[methode]['sheduler']:
                scheduler.step()
            """
        else:
            with torch.no_grad():
                model.eval()
                loss, denoised = calculate_loss(noise_images, model, batch_idx)
        #calculate psnr
        if modi == 2:
            denoised = denoised*std
        max_intensity=clean.max()-clean.min()
        if modi == 3:
            max_intensity=denoised.max()-denoised.min()
        mse = torch.mean((clean-denoised)**2)
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)
        #calculatte scaled variance (https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277/1?file=26674421) In[13]
        sv = torch.mean((noise_images/std - denoised)**2, dim=-1) / torch.mean((noise_images/std)**2, dim=-1)
        sv = torch.mean(sv)#, dim=-1)

        #log data
        psnr_log.append(round(psnr.item(),3))
        loss_log.append(loss.item())
        scaledVariance_log.append(round(sv.item(),3))
        writer.add_scalar(f'Sigma {mode}', noise.std(), global_step=epoch * len(dataLoader) + batch_idx)
        #show picture
        if psnr > bestPsnr + 0.5:
            if psnr > bestPsnr:
                bestPsnr = psnr
            saveAndPicture(psnr.item(), clean, noise_images, denoised, mode, writer, epoch, len(dataLoader), batch_idx, model, store_path)
    return loss_log, psnr_log, scaledVariance_log, bestPsnr

def main(arggv):
    print("Starte Programm!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    
    strain_train_dir = "data/DAS/SIS-rotated_train_50Hz.npy"
    strain_test_dir = "data/DAS/SIS-rotated_train_50Hz.npy"

    print("lade Datensätze ...")
    eq_strain_rates = np.load(strain_train_dir)
    split_idx = int(0.8 * len(eq_strain_rates))
    eq_strain_rates_train = torch.tensor(eq_strain_rates[:split_idx])
    eq_strain_rates_val = torch.tensor(eq_strain_rates[split_idx:])
    eq_strain_rates_test = np.load(strain_test_dir)
    eq_strain_rates_test = torch.tensor(eq_strain_rates_test)
    dataset = SyntheticNoiseDAS(eq_strain_rates_train, nx=dasChanelsTrain, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=3424, mode="train")
    dataset_validate = SyntheticNoiseDAS(eq_strain_rates_val, nx=dasChanelsVal, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=640, mode="val")
    dataset_test = SyntheticNoiseDAS(eq_strain_rates_test, nx=dasChanelsTest, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=640, mode="test")

    store_path_root = log_files()
    global modi
    for i in range(4):

        store_path = Path(os.path.join(store_path_root, f"n2self-{modi}"))
        store_path.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "tensorboard"))
        tmp.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "models"))
        tmp.mkdir(parents=True, exist_ok=True)

        print("n2self")
        dataLoader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=batchsize, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=False)

        model = U_Net(1, first_out_chanel=4, scaling_kernel_size=(1,4), conv_kernel=(3,5), batchNorm=batchnorm).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        """
        if method_params['sheduler']:
            lr_lambda = get_lr_lambda(method_params['lr'], method_params['changeLR_steps'], method_params['changeLR_rate'])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        """
        writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))

        bestPsnrTrain=0
        bestPsnrVal=0
        bestPsnrTest=0
        for epoch in tqdm(range(epochs)):

            loss, psnr, scaledVariance_log, bestPsnrTrain = train(model, device, dataLoader, optimizer, mode="train", writer=writer, epoch=epoch, store_path=store_path, bestPsnr=bestPsnrTrain)
            for i, loss_item in enumerate(loss):
                writer.add_scalar('Loss Train', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('PSNR Train', psnr[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Scaled Variance Train', scaledVariance_log[i], epoch * len(dataLoader) + i)

            loss_val, psnr_val, scaledVariance_log_val, bestPsnrVal = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch, store_path=store_path, bestPsnr=bestPsnrVal) 
            for i, loss_item in enumerate(loss_val):
                writer.add_scalar('Loss Val', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('PSNR Val', psnr_val[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Scaled Variance Val', scaledVariance_log_val[i], epoch * len(dataLoader) + i)

            if epoch % 5 == 0  or epoch==epochs-1:
                model_save_path = os.path.join(store_path, "models", f"{epoch}-model.pth")
                if save_model:
                    torch.save(model.state_dict(), model_save_path)
                else:
                    f = open(model_save_path, "x")
                    f.close()

        loss_test, psnr_test, scaledVariance_log_test, bestPsnrTest = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=0, store_path=store_path, bestPsnr=bestPsnrTest)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 0)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 0)
        writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 0)
        modi += 1

if __name__ == '__main__':
    app.run(main)