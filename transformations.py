import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm


def generate_patches_from_list(data, num_patches_per_img=None, shape=(64, 64), augment=True, shuffle=False):
        """
        Extracts patches from 'list_data', which is a list of images, and returns them in a 'numpy-array'. The images
        can have different dimensionality.

        Parameters
        ----------
        data                : list(array(float))
                            List of images with dimensions 'SZYXC' or 'SYXC'
        num_patches_per_img : int, optional(default=None)
                            If 'None', as many patches as fit i nto the dimensions are extracted.
                            Else may generate overlapping patches
        shape               : tuple(int), optional(default=(256, 256))
                            Shape of the extracted patches.
        augment             : bool, optional(default=True)
                            Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.
        shuffle             : bool, optional(default=False)
                            Shuffles extracted patches across all given images (data).

        Returns
        -------
        patches : array(float)
                Numpy-Array with the patches. The dimensions are 'SZYXC' or 'SYXC'
        """
        patches = []
        for img in data:
            patches.append( generate_patches(img.unsqueeze(0), num_patches=num_patches_per_img, shape=shape, augment=augment) )
        patches = torch.cat(patches, dim=0)

        if shuffle:
            indices = torch.randperm(len(patches))
            patches = patches[indices]
        return patches

def generate_patches(data, num_patches=None, shape=(64, 64), augment=True):
    """
    Extracts patches from 'data'. The patches can be augmented, which means they get rotated three times
    in XY-Plane and flipped along the X-Axis. Augmentation leads to an eight-fold increase in training data.

    Parameters
    ----------
    data        : list(array(float))
                List of images with dimensions 'SZYXC' or 'SYXC'
    num_patches : int, optional(default=None)
                Number of patches to extract per image. If 'None', as many patches as fit into the
                dimensions are extracted.
    shape       : tuple(int), optional(default=(256, 256))
                Shape of the extracted patches.
    augment     : bool, optional(default=True)
                Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.

    Returns
    -------
    patches : array(float)
            Numpy-Array containing all patches (randomly shuffled along S-dimension).
            The dimensions are 'SZYXC' or 'SYXC'
    """
    patches = __extract_patches__(data, num_patches=num_patches, shape=shape)
    if augment and shape[0] == shape[1]:
        patches = __augment_patches__(patches)

    if num_patches is not None:
        indices = torch.randint(len(patches), (num_patches,))
        patches = patches[indices]

    print('Generated patches:', patches.shape)
    return patches
    
def __extract_patches__(data, num_patches=None, shape=(64, 64)):
    patches = []
    if num_patches is None:
        if data.shape[-2] >= shape[0] and data.shape[-1] >= shape[1]:
            for y in range(0, data.shape[-2] - shape[0] + 1, shape[0]):
                for x in range(0, data.shape[-1] - shape[1] + 1, shape[1]):
                    patches.append(data[..., y:y + shape[0], x:x + shape[1]])
    else:
        for i in range(num_patches):
            y, x = torch.randint(0, data.shape[-2] - shape[0] + 1, (2,))
            patches.append(data[..., y:y + shape[0], x:x + shape[1]])
    return torch.cat(patches, axis=0)

def __augment_patches__(patches):
    augmented = torch.cat((patches,
                            torch.rot90(patches, 1, (-2, -1)),
                            torch.rot90(patches, 2, (-2, -1)),
                            torch.rot90(patches, 3, (-2, -1))),
                            dim=0)
    augmented = torch.cat((augmented, torch.flip(augmented, [-2])))
    return augmented




def calculate_mean_std_dataset(dataset, sigma, device, mean=None, std=None):
    #wenn Trainingsdaten
    if mean==None:
        dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)
        mean = torch.zeros(3).to(device)#RGB
        std = torch.zeros(3).to(device)
        total_samples = 0

        for batch_idx, (original, label) in enumerate(tqdm(dataLoader)):
            original = original.to(device)
            batch_size = original.size(0)
            total_samples += batch_size
            mean += original.mean(dim=(0, 2, 3))
            std += original.std(dim=(0, 2, 3))

        mean = mean / total_samples
        std = std / total_samples
        print(mean)
        print(std)

    transform_noise = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * sigma),  #Rauschen
        transforms.Normalize(mean=mean, std=std) #Normaalisieren
        ])

    #dataLoader = DataLoader(dataset, batch_size=64, shuffle=True, transform=transform_noise)
    return mean, std

def calcullate_min_max(dataset):
    min_v = None
    max_v = None
    dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch_idx, (original, label) in enumerate(tqdm(dataLoader)):
        memory = original.min()
        if min_v > memory:
            min_v = memory
        memory = original.max()
        if max_v < memory:
            max_v = memory
    return min_v, max_v
