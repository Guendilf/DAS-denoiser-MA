import torch
import numpy as np


class Mask:
    def cut2self_mask(image_size, batch_size, mask_size=(4, 4), mask_percentage=0.003): #TODO: random maskieren
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

        return torch.stack(masks, dim=0), num_regions



    def n2self_mask(noise_image, i, grid_size=3, mode='interpolate'):
        phasex = i % grid_size
        phasey = (i // grid_size) % grid_size
        mask = n2self_pixel_grid_mask(noise_image[0, 0].shape, grid_size, phasex, phasey)
        mask = mask.to(noise_image.device)
        #mask_inv = torch.ones_like(mask) - mask
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
    

    def n2void_mask(image_shape, num_masked_pixels=8):
        """
        uniform_pixel_selection_mask
        Erstellt eine Uniform Pixel Selection Maske.
        
        image_shape (tuple): Die Form des Bildes (batch, channels, height, width).
        num_masked_pixels (int): Die Anzahl der maskierten Pixel, die ausgewählt werden sollen.
        """
        if len(image_shape)==3:
            return select_random_pixels(image_shape, num_masked_pixels)
        else:
            mask_for_batch = []
            for i in range(image_shape[0]):
                mask_for_batch.append(select_random_pixels((image_shape[1],image_shape[2],image_shape[3]), num_masked_pixels))
            return torch.stack(mask_for_batch)
    
    def exchange_in_mask_with_pixel_in_window(mask, data, windowsize, num_masked_pixels):
        """
        ersetzt die ausgewählten Pixel durch die Maske mit einem zufälligem Pixel in der Fenstergröße.
        Zentrum des Fensters ist das Pixel
        
        mask (tensor): Die benutzte Makse (batch, channels, height, width).
        data (tensor): Das benutzte Bild für die ersetzung der Pixel (batch, channels, height, width)
        windowsize (int): Quadratische Fenstergröße meistens 5x5
        num_masked_pixels (int): Die Anzahl der maskierten Pixel, die ausgewählt werden sollen.
        """
        cords = torch.nonzero(mask)
        bearbeittete_Bilder = data.clone()
        memory = []
        for pixel_idx in range(cords.shape[0]): #cords.shape=(batch*num_mask_pixel*chanel, 4)
            batch, chanel, x, y = cords[pixel_idx]
            batch, chanel, x, y = batch.item(), chanel.item(), x.item(), y.item()
            if chanel != 0:
                bearbeittete_Bilder[batch, chanel, x, y] = data[batch, chanel, memory[pixel_idx%num_masked_pixels][0], memory[pixel_idx%num_masked_pixels][1]]
                if chanel==3 and (pixel_idx%num_masked_pixels)==(num_masked_pixels-1):
                    memory = []
            else: 
                new_x = max(0, min(bearbeittete_Bilder.shape[2] - 1, x + torch.randint(-windowsize//2, windowsize//2 + 1, (1,)).item()))
                new_y = max(0, min(bearbeittete_Bilder.shape[2] - 1, y + torch.randint(-windowsize//2, windowsize//2 + 1, (1,)).item()))
                memory.append((new_x, new_y))
                bearbeittete_Bilder[batch, chanel, x, y] = data[batch, chanel, new_x, new_y]
        return bearbeittete_Bilder
        

    
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

#TODO: wann benutzen?
def n2self_infer_full_image(noise_image, model, n_masks=3): #n_masks=grid_size
    net_input, mask = Mask.n2self_mask(noise_image, 0)
    net_output = model(net_input)
    acc_tensor = torch.zeros(net_output.shape).cpu()
    for i in range(n_masks):
        net_input, mask = Mask.n2self_mask(noise_image, i)
        net_output = model(net_input)
        acc_tensor = acc_tensor + (net_output * mask).cpu()
    return acc_tensor