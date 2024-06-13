from skimage.metrics import structural_similarity as sim
import statistics
import numpy as np
import torch
import torch.nn.functional as F

class Metric:
    def calculate_psnr(x,y):
        max_intensity = 1.0  # weil Vild [0, 1]
        if torch.is_tensor(x):
            if torch.is_tensor(y):
                mse = torch.mean((x-y)**2)
                psnr = 10 * torch.log10((max_intensity ** 2) / mse)
        else:
            mse = np.mean((x-y)**2)
            psnr = 10 * np.log10((max_intensity ** 2) / mse)
        return psnr

    def calculate_similarity(image1, image2):
        image1 = image1.cpu().detach().numpy()
        image2 = image2.cpu().detach().numpy()
        image_range = image1.max() - image1.min()
        similarity_index, diff_image = sim(image1, image2, data_range=image_range, channel_axis=1, full=True)
        return similarity_index, diff_image
    
    def tv_norm(x):
        """
        Args:
            x: torch tensor to calculate its total variation
        Return:
            avarge total variation (avarge tv for eevery pixel)
        """
        h_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
        w_diff = x[:, :, :, 1:] - x[:, :, :, :-1]

        h_diff_abs = torch.abs(h_diff)
        w_diff_abs = torch.abs(w_diff)

        tv_norm = torch.sum(h_diff_abs) + torch.sum(w_diff_abs)
        #durchschnitt bilden
        return tv_norm / (np.prod(x.shape))
    



    def ssim(img1, img2, window_size=11, size_average=True):
        """implementation von strukturelle Ähnlichkeitsmaß (SSIM) von noise2Self https://github.com/czbiohub-sf/noise2self/blob/master/util.py
        """
        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, window_size, channel, size_average)
    
    def avg_list(list):
        return statistics.mean(list)
    





def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)