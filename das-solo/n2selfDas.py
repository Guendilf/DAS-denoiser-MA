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
from unet_copy import UNet as unet
from import_files import SyntheticNoiseDAS
from scipy import signal

epochs = 2000 #2.000 epochen - 1 Epoche = 3424 samples
batchsize = 32
dasChanelsTrain = 11
dasChanelsVal = 11
dasChanelsTest = 11
lr = 0.0001
batchnorm = False
save_model = False

gauge_length = 19.2 #30 for synthhetic?
snr = (-2,4) #(np.log(0.01), np.log(10)) for synthhetic?
slowness = (1/10000, 1/200) #(0.2*10**-3, 10*10**-3) #angabe in m/s, laut paper 0.2 bis 10 km/s     defaault: # (0.0001, 0.005)

modi = 0 #testing diffrent setups


"""
TODO:
- swuish activation funktion with ??? parameter     -> Sigmoid Linear Unit (SiLU) function
- gauge = 19,2
- 50 Hz frequenz
"""
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

def save_das_graph(clean, noise_image, denoised):
    if isinstance(clean, torch.Tensor):
        clean = clean.to('cpu').detach().numpy()
    if isinstance(noise_image, torch.Tensor):
        noise_image = noise_image.to('cpu').detach().numpy()
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.to('cpu').detach().numpy()
    # Create a figure with 2 rows and 4 columns
    fig, axes = plt.subplots(clean.shape[0], 4, figsize=(20, 5*clean.shape[0]))

    # Plot the waves
    for i in range(clean.shape[0]):
        y_min = clean[i].min()
        y_max = clean[i].max()
        y_abs = max(abs(y_min), abs(y_max))
        # Clean wave
        axes[i, 0].plot(clean[i, 0, 0, :], label='Clean')
        axes[i, 0].set_title(f'Clean Wave {i+1}')
        axes[i, 0].set_ylim(-y_abs, y_abs)
        
        # Denoised wave
        axes[i, 1].plot(denoised[i, 0, 0, :], label='Reconstructed')
        axes[i, 1].set_title(f'Reconstructed Wave {i+1}')
        axes[i, 1].set_ylim(-y_abs, y_abs)
        
        # Noise wave
        axes[i, 2].plot(noise_image[i, 0, 0, :], label='Input')
        axes[i, 2].set_title(f'Input Wave {i+1}')
        axes[i, 2].set_ylim(-y_abs, y_abs)
        
        # Overlapping clean and denoised waves
        axes[i, 3].plot(clean[i, 0, 0, :], label='Clean', color='black')
        axes[i, 3].plot(denoised[i, 0, 0, :], label='Reconstructed', color='red')
        axes[i, 3].set_ylim(-y_abs, y_abs)
        axes[i, 3].set_title(f'Clean and Denoised comparison {i+1}')
        axes[i, 3].legend()

    # Ensure the scales in the subplots are the same
    for ax in axes.flat:
        ax.label_outer()
    plt.tight_layout()
    return fig

def save_das_imshow(images, titles):
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        image = image.to('cpu').detach().numpy()
        axs[i].imshow(image, origin='lower', aspect='auto', cmap='seismic', vmin=-1, vmax=1) #cmap='viridis'
        axs[i].set_title(title)
        axs[i].axis('off')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Bild als Tensor in TensorBoard speichern
    image_tensor = torch.Tensor(np.array(plt.imread(buf)))
    image_tensor = image_tensor.permute(2, 0, 1)  # Channels First (C, H, W)
    return image_tensor, fig
    
def saveAndPicture(psnr, clean, noise_images, denoised, mask, mode, writer, epoch, len_dataloader, batch_idx, model, store, best):
    #imshow
    noise_images_mask = noise_images * mask
    denoised_mask = denoised * mask
    images = [clean[0, 0, :, :], denoised[0, 0, :, :], noise_images[0, 0, :, :], noise_images_mask[0, 0, :, :], denoised_mask[0, 0, :, :]]
    titles = ['Clean', 'Denoised', 'Input', 'Input * Mask', 'Denoised * Mask']
    image_imshow, imshow_fig = save_das_imshow(images, titles)
    #plt.show()
    plt.close(imshow_fig)

    #graphen
    clean = clean[:2]
    clean = clean[:,:,:,0:512]
    noise_images = noise_images[:2]
    noise_images = noise_images[:,:,:,0:512]
    denoised = denoised[:2]
    denoised = denoised[:,:,:,0:512]
    chanels = []
    for i in range(clean.shape[0]):
        for j in range(clean.shape[2]):
            if mask[i,0,j,0] == 1:
                chanels.append(j)
                break
    clean = clean[:,:,chanels,:]
    noise_images = noise_images[:,:,chanels,:]
    denoised = denoised[:,:,chanels,:]
    fig = save_das_graph(clean, noise_images, denoised)
    # Speichere das Bild in TensorBoard
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    #plt.show()
    plt.close(fig)
    buf.seek(0)
    image_graph = np.array(Image.open(buf))

    

    if mode == "train":
        #writer.add_image('Denoised Training', grid, global_step=epoch * len_dataloader + batch_idx)
        writer.add_image('Graph Denoised Training', image_graph, global_step=epoch * len_dataloader + batch_idx, dataformats='HWC')
        writer.add_image('Imshow Denoised Training', image_imshow, global_step=epoch * len_dataloader + batch_idx)
    elif mode == "val":
        #writer.add_image('Denoised Validation', grid, global_step=epoch * len_dataloader + batch_idx)
        writer.add_image('Graph Denoised Validation', image_graph, global_step=epoch * len_dataloader + batch_idx, dataformats='HWC')
        writer.add_image('Imshow Denoised Validation', image_imshow, global_step=epoch * len_dataloader + batch_idx)
    else:
        #writer.add_image('Denoised Test', grid, global_step=1 * len_dataloader + batch_idx)
        writer.add_image('Graph Denoised Test', image_graph, global_step=epoch * len_dataloader + batch_idx, dataformats='HWC')
        writer.add_image('Imshow Denoised Test', image_imshow, global_step=epoch * len_dataloader + batch_idx)
    #TODO:
    #imshow(denoised) und co aspecratio, vmin, vmax
    """
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
    """

def calculate_loss(noise_image, model, batch_idx):
    #masked_noise_image, mask = Mask.n2self_mask(noise_image, batch_idx)
    mask = torch.zeros_like(noise_image).to(noise_image.device)
    for i in range(mask.shape[0]):
        mask[i, :, np.random.randint(0, mask.shape[2]), :] = 1
    masked_noise_image = (1-mask) * noise_image
    denoised = model(masked_noise_image)
    return torch.nn.MSELoss()(denoised*(mask), noise_image*(mask)), denoised, mask

def train(model, device, dataLoader, optimizer, mode, writer, epoch, store_path, bestPsnr):
    global modi
    loss_log = []
    psnr_log = []
    scaledVariance_log = []
    save_on_last_epoch=True
    for batch_idx, (noise_images, clean, noise, std, amp, _, _, _) in enumerate(dataLoader):
        clean = clean.to(device).type(torch.float32)
        noise_images = noise_images.to(device).type(torch.float32)
        std = std.to(device).type(torch.float32)
        if mode == "train":
            model.train()
            loss, denoised, mask_orig = calculate_loss(noise_images, model, batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                loss, denoised, mask_orig = calculate_loss(noise_images, model, batch_idx)
                buffer = torch.zeros_like(denoised).to(device)
                for i in range(denoised.shape[2]):
                    mask = torch.zeros_like(clean).to(device)
                    #11 because i use 11 DAS Chanels during training
                    mask[:,:,i%11,:] = 1
                    input_image = noise_images[:,:,int(i/11):int(i/11)+11,:] * (1-mask)
                    j_denoised = model(input_image)
                    buffer += j_denoised*mask
                denoised = buffer
        #calculate psnr
        max_intensity=clean.max()-clean.min()
        mse = torch.mean((clean-denoised)**2)
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)
        #calculatte scaled variance (https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277/1?file=26674421) In[13]
        sv = torch.mean((noise_images - denoised)**2, dim=-1) / torch.mean((noise_images)**2, dim=-1)
        sv = torch.mean(sv)#, dim=-1)

        #log data
        psnr_log.append(round(psnr.item(),3))
        loss_log.append(loss.item())
        scaledVariance_log.append(round(sv.item(),3))
        writer.add_scalar(f'Sigma {mode}', noise.std(), global_step=epoch * len(dataLoader) + batch_idx)
        if batch_idx % 50 == 0 or batch_idx == len(dataLoader)-1:
            saveAndPicture(psnr.item(), clean, noise_images, denoised, mask_orig, mode, writer, epoch, len(dataLoader), batch_idx, model, store_path, True)
        elif mode == "test" and batch_idx%50 == 0:
            saveAndPicture(psnr.item(), clean, noise_images, denoised, mask_orig, mode, writer, epoch, len(dataLoader), batch_idx, model, store_path, True)
        """
        if psnr > bestPsnr + 0.5:
            if psnr > bestPsnr:
                bestPsnr = psnr
            saveAndPicture(psnr.item(), clean, noise_images, denoised, mask_orig, mode, writer, epoch, len(dataLoader), batch_idx, model, store_path, True)
            if batch_idx >= len(dataLoader)-5:
                save_on_last_epoch = False
    if save_on_last_epoch:
        saveAndPicture(psnr.item(), clean, noise_images, denoised, mask_orig, mode, writer, epoch, len(dataLoader), batch_idx, model, store_path, False)
    """
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
    dataset = SyntheticNoiseDAS(eq_strain_rates_train, nx=dasChanelsTrain, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=10016, mode="train")
    dataset_validate = SyntheticNoiseDAS(eq_strain_rates_val, nx=dasChanelsVal, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=992, mode="val")
    dataset_test = SyntheticNoiseDAS(eq_strain_rates_test, nx=dasChanelsTest, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=992, mode="test")

    store_path_root = log_files()
    global modi
    for i in range(2):

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

        if modi == 0:
            model = U_Net(1, first_out_chanel=4, scaling_kernel_size=(1,4), conv_kernel=(3,5), batchNorm=batchnorm, n2self_architecture=True).to(device)
            #model = unet(n_channels=1, feature=4, bilinear=True).to(device)
        else:
            model = U_Net(1, first_out_chanel=4, scaling_kernel_size=(1,4), conv_kernel=(3,5), batchNorm=batchnorm, n2self_architecture=False).to(device)
            #model = unet(n_channels=1, feature=4, bilinear=False).to(device)
        
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

            writer.add_scalar('Loss Train', statistics.mean(loss), epoch)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch)

            loss_val, psnr_val, scaledVariance_log_val, bestPsnrVal = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch, store_path=store_path, bestPsnr=bestPsnrVal) 

            writer.add_scalar('Loss Val', statistics.mean(loss_val), epoch)
            writer.add_scalar('PSNR Val', statistics.mean(psnr_val), epoch)
            writer.add_scalar('Scaled Variance Val', statistics.mean(scaledVariance_log_val), epoch)

            if epoch % 5 == 0  or epoch==epochs-1:
                model_save_path = os.path.join(store_path, "models", f"{epoch}-model.pth")
                if save_model  or epoch==epochs-1:
                    torch.save(model.state_dict(), model_save_path)
                else:
                    f = open(model_save_path, "x")
                    f.close()

        loss_test, psnr_test, scaledVariance_log_test, bestPsnrTest = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=0, store_path=store_path, bestPsnr=bestPsnrTest)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 0)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 0)
        writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 0)
        model_save_path = os.path.join(store_path, "models", "last-model.pth")
        torch.save(model.state_dict(), model_save_path)
        modi += 1
    print("fertig")

if __name__ == '__main__':
    app.run(main)