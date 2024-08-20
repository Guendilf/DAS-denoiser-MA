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
import io
from PIL import Image

from import_files import log_files
from import_files import U_Net
from import_files import n2nU_net
from import_files import SyntheticNoiseDAS
from scipy import signal

epochs = 30 #2.000 epochen - 1 Epoche = 3424 samples
batchsize = 24
dasChanels = 96
timesamples = 128 #oder 30 sekunden bei samplingrate von 100Hz -> ?
lr = 0.01
lr_end = 0.0001
# Define the lambda function for the learning rate scheduler
lambda_lr = lambda epoch: lr_end + (lr - lr_end) * (1 - epoch / epochs)

batchnorm = False
save_model = False

snr_level = log_SNR=(-2,4)#default, ist weas anderes
gauge_length = 1 #30 for synthhetic?
snr = 2#(-2,4) #(np.log(0.01), np.log(10)) for synthhetic?
slowness = (1/10000, 1/200) #(0.2*10**-3, 10*10**-3) #angabe in m/s, laut paper 0.2 bis 10 km/s     defaault: # (0.0001, 0.005)

modi = 0 #testing diffrent setups


def show_das(original, norm=True):
    if isinstance(original, torch.Tensor):
        original = original.to('cpu').detach().numpy()
    original = original[0]
    plt.figure(figsize=(7, 5))
    for i in range(original.shape[1]):
        if norm:
            std = original[0][i].std()
            if std == 0:
                std = 0.000000001
        sr = original[0][i]
        plt.plot(sr + 3*i, c="k", lw=0.5, alpha=1)
        #if every chanle by it self
        #plt.subplot(original.shape[1], 1, i + 1)
        #plt.plot(original[0, i].numpy(), c="k", lw=0.5, alpha=1)
        #plt.title(f'Channel {i + 1}')
        plt.tight_layout()
    plt.show()

def save_das_graph(original, denoised, noise):
    def plot_das(data, title, ax, batch_idx):
        if isinstance(data, torch.Tensor):
            data = data.to('cpu').detach().numpy()
        data = data[batch_idx]
        for i in range(data.shape[1]):
            #std = data[0][i].std()
            #if std == 0:
                #std = 0.000000001
            sr = data[0][i]# / std
            #sr = data[0][i]
            ax.plot(sr + 3*i, c="k", lw=0.5, alpha=1)
        ax.set_title(title)
        ax.set_axis_off()
    
    # Erstelle eine Figur mit 3 Subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))

    # Erste Spalte - Batch 0
    plot_das(original, 'Original (Clean) 1', axs[0, 0], 0)
    plot_das(denoised, 'Reconstructed 1', axs[1, 0], 0)
    plot_das(noise, 'Input (with noise) 1', axs[2, 0], 0)
    # Zweite Spalte - Batch 1
    plot_das(original, 'Original (Clean) 2', axs[0, 1], 1)
    plot_das(denoised, 'Reconstructed 2', axs[1, 1], 1)
    plot_das(noise, 'Noise (with noise) 2', axs[2, 1], 1)
    plt.tight_layout()
    
    return fig
    
def saveAndPicture(psnr, clean, noise_images, denoised, mode, writer, epoch, len_dataloader, batch_idx, model, store, best):
    #comparison = torch.cat((clean[:1], denoised[:1], noise_images[:1]), dim=0)
    #comparison = comparison[:,:,:,:512]
    #grid = make_grid(comparison, nrow=1, normalize=False).cpu()

    clean = clean[:2]
    clean = clean[:,:,:,0:512]
    noise_images = noise_images[:2]
    noise_images = noise_images[:,:,:,0:512]
    denoised = denoised[:2]
    denoised = denoised[:,:,:,0:512]
    fig = save_das_graph(clean, denoised, noise_images)
    # Speichere das Bild in TensorBoard
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    #plt.show()
    plt.close(fig)
    buf.seek(0)
    image = np.array(Image.open(buf))

    if mode == "train":
        #writer.add_image('Denoised Training', grid, global_step=epoch * len_dataloader + batch_idx)
        writer.add_image('Denoised Training', image, global_step=epoch * len_dataloader + batch_idx, dataformats='HWC')
    elif mode == "validate":
        #writer.add_image('Denoised Validation', grid, global_step=epoch * len_dataloader + batch_idx)
        writer.add_image('Denoised Validation', image, global_step=epoch * len_dataloader + batch_idx, dataformats='HWC')
    else:
        #writer.add_image('Denoised Test', grid, global_step=1 * len_dataloader + batch_idx)
        writer.add_image('Denoised Test', image, global_step=epoch * len_dataloader + batch_idx, dataformats='HWC')
    if not best:
        return
    if "test" not in mode:
        print(f"best model found with psnr: {psnr}")
        model_save_path = os.path.join(store, "models", f"{round(psnr, 1)}-{mode}-{epoch}-{batch_idx}.pth")
        if save_model:
            torch.save(model.state_dict(), model_save_path)
        else:
            f = open(model_save_path, "x")
            f.close()

def calculate_loss(noise_image, noise_image2, model):
    denoised = model(noise_image)
    loss = 1/noise_image.shape[0] * torch.sum((denoised-noise_image2)**2)
    return loss, denoised

def train(model, device, dataLoader, optimizer, scheduler, mode, writer, epoch, store_path, bestPsnr):
    global modi
    loss_log = []
    psnr_log = []
    scaledVariance_log = []
    save_on_last_epoch=True
    for batch_idx, (noise_images, clean, noise, std, amp, noise_images2, noise2, std2) in enumerate(dataLoader):
        clean = clean.to(device).type(torch.float32)
        noise_images = noise_images.to(device).type(torch.float32)
        noise_images2 = noise_images2.to(device).type(torch.float32)
        std = std.to(device).type(torch.float32)
        std2 = std2.to(device).type(torch.float32)
        if mode == "train":
            model.train()
            loss, denoised = calculate_loss(noise_images, noise_images2, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            """
            if config.methodes[methode]['sheduler']:
                scheduler.step()
            """
            writer.add_scalar('Learning Rate N2N DAS', scheduler.get_last_lr()[0], global_step=epoch * len(dataLoader) + batch_idx)
        else:
            with torch.no_grad():
                model.eval()
                loss, denoised = calculate_loss(noise_images, noise_images2, model)
        #calculate psnr
        max_intensity=clean.max()-clean.min()
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
        if psnr > bestPsnr + 0.5:
            if psnr > bestPsnr:
                bestPsnr = psnr
            saveAndPicture(psnr.item(), clean, noise_images, denoised, mode, writer, epoch, len(dataLoader), batch_idx, model, store_path, True)
            if batch_idx >= len(dataLoader)-5:
                save_on_last_epoch = False
    if save_on_last_epoch:
        saveAndPicture(psnr.item(), clean, noise_images, denoised, mode, writer, epoch, len(dataLoader), batch_idx, model, store_path, False)
    return loss_log, psnr_log, scaledVariance_log, bestPsnr

def main(arggv):
    global dasChanels
    global timesamples
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
    dataset = SyntheticNoiseDAS(eq_strain_rates_train, nx=dasChanels, nt=timesamples, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, fs=1000.0, size=2568, mode="train")
    dataset_validate = SyntheticNoiseDAS(eq_strain_rates_val, nx=dasChanels, nt=timesamples, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, fs=1000.0, size=480, mode="val")
    dataset_test = SyntheticNoiseDAS(eq_strain_rates_test, nx=dasChanels, nt=timesamples, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, fs=1000.0, size=480, mode="test")

    store_path_root = log_files()
    global modi
    for i in range(5):
        
        store_path = Path(os.path.join(store_path_root, f"n2noise-{modi}"))
        store_path.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "tensorboard"))
        tmp.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "models"))
        tmp.mkdir(parents=True, exist_ok=True)
        """
        if modi==1:
            dasChanels = 11
            timesamples = 2048
            dataset = SyntheticNoiseDAS(eq_strain_rates_train, nx=dasChanels, nt=timesamples, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=2568, mode="train")
            dataset_validate = SyntheticNoiseDAS(eq_strain_rates_val, nx=dasChanels, nt=timesamples, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=480, mode="val")
            dataset_test = SyntheticNoiseDAS(eq_strain_rates_test, nx=dasChanels, nt=timesamples, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=480, mode="test")
        """
        print("n2noise")
        dataLoader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=batchsize, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=False)
        if modi==0:
            model = n2nU_net(1, first_out_chanel=24, scaling_kernel_size=2, conv_kernel=3, batchNorm=batchnorm).to(device)
        elif modi==1:
            model = n2nU_net(1, first_out_chanel=24, scaling_kernel_size=2, conv_kernel=3, batchNorm=True).to(device)
        elif modi==2:
            model = n2nU_net(1, first_out_chanel=24, scaling_kernel_size=(1,2), conv_kernel=3, batchNorm=batchnorm).to(device)
        elif modi==3:
            model = n2nU_net(1, first_out_chanel=24, scaling_kernel_size=(1,2), conv_kernel=3, batchNorm=True).to(device)
        else:
            model = U_Net(1, first_out_chanel=24, scaling_kernel_size=(1,2), conv_kernel=5, batchNorm=batchnorm).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

        writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))

        bestPsnrTrain=0
        bestPsnrVal=0
        bestPsnrTest=0
        for epoch in tqdm(range(epochs)):

            loss, psnr, scaledVariance_log, bestPsnrTrain = train(model, device, dataLoader, optimizer, scheduler, mode="train", writer=writer, epoch=epoch, store_path=store_path, bestPsnr=bestPsnrTrain)
            """
            for i, loss_item in enumerate(loss):
                writer.add_scalar('Loss Train', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('PSNR Train', psnr[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Scaled Variance Train', scaledVariance_log[i], epoch * len(dataLoader) + i)
            """
            writer.add_scalar('Loss Train', statistics.mean(loss), epoch)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch)

            loss_val, psnr_val, scaledVariance_log_val, bestPsnrVal = train(model, device, dataLoader_validate, optimizer, scheduler, mode="val", writer=writer, epoch=epoch, store_path=store_path, bestPsnr=bestPsnrVal) 
            """
            for i, loss_item in enumerate(loss_val):
                writer.add_scalar('Loss Val', loss_item, epoch * len(dataLoader) + i)
                writer.add_scalar('PSNR Val', psnr_val[i], epoch * len(dataLoader) + i)
                writer.add_scalar('Scaled Variance Val', scaledVariance_log_val[i], epoch * len(dataLoader) + i)
            """
            writer.add_scalar('Loss Val', statistics.mean(loss_val), epoch)
            writer.add_scalar('PSNR Val', statistics.mean(psnr_val), epoch)
            writer.add_scalar('Scaled Variance Val', statistics.mean(scaledVariance_log_val), epoch)

            if epoch % 5 == 0  or epoch==epochs-1:
                model_save_path = os.path.join(store_path, "models", f"{epoch}-model.pth")
                if save_model:
                    torch.save(model.state_dict(), model_save_path)
                else:
                    f = open(model_save_path, "x")
                    f.close()

        loss_test, psnr_test, scaledVariance_log_test, bestPsnrTest = train(model, device, dataLoader_test, optimizer, scheduler, mode="test", writer=writer, epoch=0, store_path=store_path, bestPsnr=bestPsnrTest)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 0)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 0)
        writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 0)
        modi += 1

if __name__ == '__main__':
    app.run(main)