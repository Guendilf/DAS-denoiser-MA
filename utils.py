from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil

import torch



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




def show_logs(loss, psnr, value_loss, value_psnr, similarity):
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

def log_files():
    current_path = Path(os.getcwd())
    store_path = Path(os.path.join(current_path, "runs", f"run-{str(datetime.now()).replace(':', '-')}"))
    store_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(os.path.join(store_path, "models"))
    models_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path = Path(os.path.join(store_path, "tensorboard"))
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    for file in os.listdir(current_path):
        if file.endswith(".py"):
            print("copy: ", file)
            shutil.copyfile(os.path.join(current_path, file), os.path.join(store_path, file))
    print(f"python files are stored. Path: {store_path}")

    return store_path

def show_tensor_as_picture(img):
    img = img.cpu().detach()
    if len(img.shape)==4:
        img = img[0]
    if img.shape[0] == 3:
        img = img.permute(1,2,0)
    plt.imshow(img, interpolation='nearest')
    plt.show()

def normalize_image(image):
    if torch.is_tensor(image):
        image_min = image.min()
        image_max = image.max()
        return (image-image.min())  / (image.max() - image.min())
    else:
        print("kann nicht normalisieren, da kein Tensor")
        exit(32)

