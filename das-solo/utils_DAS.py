import numpy as np
import torch
from tqdm import tqdm
from skimage.util import view_as_windows
from scipy.signal import correlate
import scipy.ndimage
import scipy.signal


#cc-Metric
def xcorr(x, y):
    # implementation from https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277?file=26674424
    # FFT of x and conjugation
    X_bar = np.fft.rfft(x).conj()
    Y = np.fft.rfft(y)
    # Compute norm of data
    norm_x_sq = np.sum(x**2)
    norm_y_sq = np.sum(y**2)
    norm = np.sqrt(norm_x_sq * norm_y_sq)
    # Correlation coefficients
    R = np.fft.irfft(X_bar * Y) / norm
    # Return correlation coefficient
    return np.max(R)
def compute_xcorr_window(x):
    # implementation from https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277?file=26674424
    Nch = x.shape[0]
    Cxy = np.zeros((Nch, Nch)) * np.nan
    for i in range(Nch):
        for j in range(i):
            Cxy[i, j] = xcorr(x[i], x[j])
    return np.nanmean(Cxy)
def compute_moving_coherence(data, bin_size):
    """ implementation from https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277?file=26674424
    Args:
    data: DAS data with shape (N_ch, N_t)
    bin_size: Size of the moving window
    """    
    N_ch = data.shape[0]
    cc = np.zeros(N_ch)
    for i in range(N_ch):
        start = max(0, i - bin_size // 2)
        stop = min(i + bin_size // 2, N_ch)
        ch_slice = slice(start, stop)
        cc[i] = compute_xcorr_window(data[ch_slice])
    return cc

#                  (Sum[Signal(t)])**2
# Semblance = -----------------------------
#               Kanäle * Sum[Signal(t)**2]

def semblance(data, window_size=(15,25), moveout=False):
    """fast, without moveout correction
    Returns:
    torch.tensor(b,c,t)"""
    if len(data.shape) == 3:
        print("data shape needs to be (b,1,c,t) or (c,t) and not (?,c,t)")
    elif len(data.shape) == 2:
        data = data.unsqueeze(0).unsqueeze(0)
    if moveout:
        #data = moveout_correction(data, window_size)
        raise ValueError(f"moveout ist für diese Methode nicht implementiert")
    batch, _, channels, time = data.shape
    semblance_vals = torch.zeros((batch, channels - window_size[0] + 1, time - window_size[1] + 1), device=data.device)

    #more dimensions for window sliding
    unfolded = torch.nn.functional.unfold(data, kernel_size=(window_size[0], window_size[1])).view(batch, window_size[0], window_size[1], channels - window_size[0] + 1, time - window_size[1] + 1)
    sum_signals = (unfolded.sum(dim=1)**2).sum(dim=1)
    scaled_sum = (unfolded**2).sum(dim=1).sum(dim=1)
    
    semblance_vals = sum_signals / (window_size[0] * scaled_sum)
    return semblance_vals
# #implementiert von https://github.com/sachalapins/DAS-N2N/blob/main/results.ipynb
def correlate_func(x, idx, cc_thresh = 0.7):
    correlation = correlate(x[idx,:], x[(idx+1),:], mode="full")
    lags = np.arange(-(x[idx,:].size - 1), x[(idx+1),:].size)
    lag_idx = np.argmax(correlation)
    lag = lags[lag_idx]
    if lag > 0:
        if np.corrcoef(x[idx,lag:], x[(idx+1),:-lag], rowvar=False)[0,1] > cc_thresh:
            x = np.concatenate(
                [np.concatenate([x[:(idx+1),:], np.zeros((x[:(idx+1),:].shape[0], lag))], axis=1),
                 np.concatenate([np.zeros((x[(idx+1):,:].shape[0], lag)), x[(idx+1):,:]], axis=1)],
                axis=0)
    if lag < 0:
        if np.corrcoef(x[idx,:-lag], x[(idx+1),lag:], rowvar=False)[0,1] > cc_thresh:
            x = np.concatenate(
                [np.concatenate([np.zeros((x[:(idx+1),:].shape[0], abs(lag))), x[:(idx+1),:]], axis=1),
                 np.concatenate([x[(idx+1):,:], np.zeros((x[(idx+1):,:].shape[0], abs(lag)))], axis=1)],
                axis=0)
    return(x)
#implementiert von https://github.com/sachalapins/DAS-N2N/blob/main/results.ipynb
def moving_window_semblance(data, window):
    wrapped = lambda region: marfurt_semblance(region.reshape(window))
    return scipy.ndimage.generic_filter(data, wrapped, window)
# This is equal to first part of Eq 7 in https://doi.org/10.1111/1365-2478.13178 ; implementiert von https://github.com/sachalapins/DAS-N2N/blob/main/results.ipynb
def marfurt_semblance(region):
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape
    # Cross correlation and shift
    for i in (range(ntraces-1)):
        region = correlate_func(region, i, cc_thresh = 0.7)
    square_of_sums = np.sum(region, axis=0)**2
    sum_of_squares = np.sum(region**2, axis=0)
    sembl = square_of_sums.sum() / sum_of_squares.sum()
    return sembl / ntraces

def local_snr(image, window_size, moveout=False, cc_thresh=0.9):
    #berechnet die normale SNR in einem Fenster "window_size"
    # Verwende Gleitfenster, um lokale Regionen zu extrahieren
    windows = view_as_windows(image, (window_size[0], window_size[1]))
    snr_map = np.zeros((windows.shape[0], windows.shape[1]))

    # Berechne SNR in jedem Fenster
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            window = windows[i, j]
            # Falls moveout=True, wende correlate_func auf das Fenster an
            if moveout and window.shape[0] > 1:  # Nur wenn mehr als ein Kanal vorhanden ist
                for idx in range(window.shape[0] - 1):
                    window = correlate_func(window, idx, cc_thresh)  # Passe Kanäle mit correlate_func an
            # Signal- und Rauschkomponenten berechnen
            signal_energy = np.mean(np.abs(window))  # Signalenergie durch Mittelwert der Absolutwerte
            noise = np.var(window)  # Rauschen durch Varianz
            if noise != 0:
                snr_map[i, j] = 10 * np.log10(signal_energy**2 / noise**2)
            else:
                snr_map[i, j] = 0  # Vermeide Division durch 0
    return snr_map