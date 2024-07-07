import torch
import numpy as np


class Mask:
    #old_version
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
    

    #replaces cut2self_mask and n2void_mask
    def mask_random(img, maskamount, mask_size=(4,4)):
        """
        TODO: is not saturated sampling (but in if section for mask_size=(1,1)) -> the same Pixel could be chosen multiple times -> in one picture there is less then required amount of masking
        Args
            img (tensor): Noisy images in form of (b,c,w,h) only for shape extraction
            maskamount (number): float for percentage masking; int for area amount masking
            mask_size (tupel): area that will be masked (w,h)
        Return
            mask (tensor): masked pixel are set to 1 (b,c,w,h)
            masked pixel (int): pixel that should be masked
        """
        total_area = img.shape[-1] * img.shape[-2]
        #if amount of pixel shoulld be masked
        if isinstance(maskamount, int):
            mask_percentage = maskamount*1/total_area
        else:
            mask_percentage = maskamount
            maskamount = int(mask_percentage*total_area/1)
            if maskamount == 0:
                maskamount = 1
        mask_area = mask_size[0] * mask_size[1]
        num_regions = int((mask_percentage * total_area) / mask_area)
        masks = []
        #fast methode for pixel only "select_random_pixel" or even with nn.functional.dropout
        #saturated sampling ensured through "torch.randperm" in select_random_pixels
        if mask_size == (1,1):
            for _ in range(img.shape[0]):
                mask = select_random_pixels((img.shape[1], img.shape[2],img.shape[3]), maskamount)
                masks.append(mask)
            mask = torch.stack(masks, dim=0)
            return mask, torch.count_nonzero(mask)
        
        else:
            for _ in range(img.shape[0]):
                mask = torch.zeros(img.shape[-1], img.shape[-2], dtype=torch.float32)
                for _ in range(num_regions):        # generiere eine maske
                    x = torch.randint(0, img.shape[-1] - mask_size[1] + 1, (1,))
                    y = torch.randint(0, img.shape[-2] - mask_size[0] + 1, (1,))
                    mask[x:x+mask_size[0], y:y+mask_size[1]] = 1
                masks.append(mask)
        
        mask = torch.stack(masks, dim=0)
        mask = mask.unsqueeze(1)  # (b, 1, w, h)
        mask = mask.expand(-1, img.shape[1], -1, -1) # (b, 3, w, h)
        return mask, torch.count_nonzero(mask)

    def n2self_mask(noise_image, i, grid_size=3, mode='interpolate', include_mask_as_input=False):       #i= epoch % (width**2  -1)
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
    

    def n2self_jinv_recon(noise_image, model, grid_size=4, mode='interpolate', infer_single_pass=False, include_mask_as_input=False):
        return jinv_recon(noise_image, model, grid_size, mode, infer_single_pass, include_mask_as_input)
    
    #old_version
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
            if chanel != 0: #copy the same pixel as in chanel 0
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
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel)
    kernel = kernel / kernel.sum()
    kernel = np.repeat(kernel, repeats=tensor.shape[1], axis=1).to(device)
    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)
    return filtered_tensor * mask + tensor * mask_inv #TODO:verschieben in mask

def jinv_recon(noise_image, model, grid_size, mode, infer_single_pass, include_mask_as_input):
    """
    Original: infer_full_image from Noise2Self masks.py
    """
    """
    if infer_single_pass:
        if include_mask_as_input:
            net_input = torch.cat((noise_image, torch.zeros(noise_image[:, 0:1].shape).to(noise_image.device)), dim=1)
        else:
            net_input = noise_image
        net_output = model(net_input)
        return net_output
    """
    #else:
    net_input, mask = Mask.n2self_mask(noise_image, 0, grid_size=grid_size, mode=mode, include_mask_as_input=include_mask_as_input)
    net_output = model(net_input)
    acc_tensor = torch.zeros(net_output.shape).to(noise_image.device)#.cpu()
    for i in range(grid_size**2):
        net_input, mask = Mask.n2self_mask(noise_image, i, grid_size=grid_size, mode=mode, include_mask_as_input=include_mask_as_input)
        net_output = model(net_input)
        acc_tensor = acc_tensor + (net_output * mask).to(noise_image.device)#.cpu()
    return acc_tensor