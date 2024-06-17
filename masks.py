import torch
import numpy as np


class Mask:
    def cut2self_mask(image_size, batch_size, mask_size=(4, 4), mask_percentage=0.003, corrected=0): #TODO: random maskieren
        total_area = image_size[0] * image_size[1]  #w*h
        
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
        
        mask = torch.stack(masks, dim=0)
        #check if j-invariant (no region is overlapping in all batchhes)
        #TODO: ist das ein richhtigger Chheck oder kann man es auch besser machen?
        """
        if mask.sum().item() != num_regions*mask_area*batch_size:
            if corrected==5:
                raise Exception(f"Es wurden {corrected+1} mal keine j-invariant Masken erzeugt. Es werden {num_regions} Regionen die {mask_area} Pixel groß sind erstellt und diese müssen in {batch_size} Batches eindeutig sein.")
            mask, num_regions = Mask.cut2self_mask(image_size, batch_size, mask_size, mask_percentage, corrected=corrected+1)
        """
        return mask, num_regions



    def n2self_mask(noise_image, i, grid_size=3, mode='interpolate'):       #i= epoch % (width**2  -1)
        phasex = i % grid_size
        phasey = (i // grid_size) % grid_size
        mask = n2self_pixel_grid_mask(noise_image[0, 0].shape, grid_size, phasex, phasey)
        mask = mask.to(noise_image.device)
        mask_inv = 1 - mask
        if mode == 'interpolate':
            masked = n2self_interpolate_mask(noise_image, mask, mask_inv)
        elif mode == 'zero':
            masked = noise_image * mask_inv
        #else:
            #raise NotImplementedError
        #if self.include_mask_as_input:
            #net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
        #else:
        net_input = masked
        return net_input, mask
    

    def n2self_jinv_recon(noise_image, model):
        return jinv_recon(noise_image, model)
    

    def n2void_mask(image, num_masked_pixels=8):
        """
        uniform_pixel_selection_mask
        Erstellt eine Uniform Pixel Selection Maske.
        
        image (tensor): extrahhier die Form des Bildes (batch, channels, height, width).
        num_masked_pixels (int): Die Anzahl der maskierten Pixel, die ausgewählt werden sollen.
        """
        #if no batch
        if len(image.shape)==3:
            return select_random_pixels(image.shape, num_masked_pixels)
        else:
            mask_for_batch = []
            for i in range(image.shape[0]):
                mask_for_batch.append(select_random_pixels((image.shape[1],image.shape[2],image.shape[3]), num_masked_pixels))
            return torch.stack(mask_for_batch)
    
    def exchange_in_mask_with_pixel_in_window(mask, data, windowsize, num_masked_pixels, replaceWithIself=False):
        """
        ersetzt die ausgewählten Pixel durch die Maske mit einem zufälligem Pixel in der Fenstergröße.
        Zentrum des Fensters ist das Pixel
        
        mask (tensor): Die benutzte Makse (batch, channels, height, width).
        data (tensor): Das benutzte Bild für die ersetzung der Pixel (batch, channels, height, width)
        windowsize (int): Quadratische Fenstergröße meistens 5x5
        num_masked_pixels (int): Die Anzahl der maskierten Pixel, die ausgewählt werden sollen.
        replaceWithIself (bool): is it possible to replalce tthe selectted Pixel with itself
        """
        cords = torch.nonzero(mask) #in jeder Zeile ist die Koordinate eines Wertes der nicht = 0 ist (also 1 z.b.)
        bearbeitete_Bilder = data.clone()
        memory = []
        for pixel_idx in range(cords.shape[0]): #cords.shape=(batch*num_mask_pixel*chanel, 4)   geht alle gefundenen Koordinaten durch die nicht 0 sind in Maske
            batch, chanel, x, y = cords[pixel_idx]
            batch, chanel, x, y = batch.item(), chanel.item(), x.item(), y.item()
            if chanel != 0:
                new_x, new_y = memory[pixel_idx % num_masked_pixels]
                bearbeitete_Bilder[batch, chanel, x, y] = data[batch, chanel, new_x, new_y]
            else: 
                while True: 
                    #      max ( 0, min(width-1, x + rand(-window/2, window/2+1)) )
                    new_x = max(0, min(bearbeitete_Bilder.shape[2] - 1, x + torch.randint(-windowsize//2, windowsize//2 + 1, (1,)).item()))
                    new_y = max(0, min(bearbeitete_Bilder.shape[2] - 1, y + torch.randint(-windowsize//2, windowsize//2 + 1, (1,)).item()))
                    #don't replace with the same pixel
                    if (new_x, new_y) != (x,y) or replaceWithIself:
                        break
                memory.append((new_x, new_y))
                bearbeitete_Bilder[batch, chanel, x, y] = data[batch, chanel, new_x, new_y]
        return bearbeitete_Bilder
        

    
def select_random_pixels(image_shape, num_masked_pixels):
    num_pixels = image_shape[1] * image_shape[2]
    # Erzeuge zufällige Indizes für die ausgewählten maskierten Pixel
    masked_indices = torch.randperm(num_pixels)[:num_masked_pixels]
    mask = torch.zeros(image_shape[1], image_shape[2])
    # Pixel in Maske auf 1 setzen
    mask.view(-1)[masked_indices] = 1
    # Mache für alle Chanels
    mask = mask.unsqueeze(0).expand(image_shape[0], -1, -1)
    return mask
    

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
    return filtered_tensor * mask + tensor * mask_inv #TODO:verschieben in mask

def jinv_recon(noise_image, model): #infer_full_image in mask.py from n2Self
        grid_size=3 #gleicher wert wie in "Mask.n2self_mask" beim ttraining
        net_input, mask = Mask.n2self_mask(noise_image, 0, mode = 'zero')
        net_output = model(net_input)
        acc_tensor = torch.zeros(net_output.shape).to(noise_image.device)
        for i in range(grid_size**2):
            net_input, mask = Mask.n2self_mask(noise_image, i, mode = 'zero')
            net_output = model(net_input)
            acc_tensor = acc_tensor + (net_output * mask).to(noise_image.device)
        return acc_tensor