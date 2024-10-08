import matplotlib.pyplot as plt
import numpy as np
import torch


#get bottom line for time-axis
def set_bottom_line(ax, left=False):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
# delete black outline of plot
def remove_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

def generate_wave_plot(das, noisy_waves, denoised_waves, snrs):
    # Lege den Plot mit 3 Spalten und der gewünschten Anzahl an Zeilen fest
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1, 1]})

    # Gleiche Skalen für alle Plots festlegen
    min_wave = das[0].min().item()
    max_wave = das[0].max().item()

    das = das.to('cpu').detach().numpy()
    noisy_waves = noisy_waves.to('cpu').detach().numpy()
    denoised_waves = denoised_waves.to('cpu').detach().numpy()

    # 1. Spalte: Originalsignal plotten
    axs[1, 0].plot(das[0])
    axs[1, 0].set_title('Original Wave')
    axs[1, 0].set_ylim(min_wave, max_wave)  # Gleiche Skala für y-Achse
    remove_frame(axs[1, 0])  # Entferne den Rahmen
    set_bottom_line(axs[1, 0])  # Zeige nur die untere Linie
    axs[1, 0].yaxis.set_visible(False)  # Deaktiviere y-Achsen für das mittige Diagramm
    axs[1, 0].set_xlabel('Time')

    # Leere Felder in der ersten Spalte
    axs[0, 0].set_axis_off()  # Oberes Feld leer
    axs[2, 0].set_axis_off()  # Unteres Feld leer

    # 2. Spalte: Verrauschte Wellen plotten
    for i, noisy_wave in enumerate(noisy_waves):
        axs[i, 1].plot(noisy_wave[0])
        axs[i, 1].set_title(f'Input Wave (SNR={snrs[i]})')
        axs[i, 1].set_ylim(min_wave, max_wave)
        remove_frame(axs[i, 1])  # Entferne den Rahmen
        if i < 2:  # Nur im untersten Plot eine x-Achse anzeigen
            axs[i, 1].set_xticks([])
        else:# Für den untersten Plot die Linie und die Beschriftung anzeigen
            set_bottom_line(axs[i, 1])
            axs[i, 1].set_xlabel('Time')
        axs[i, 1].set_yticks([])

    # 3. Spalte: Entstaubte Wellen durch das Modell plotten
    for i, denoised_wave in enumerate(denoised_waves):
        axs[i, 2].plot(denoised_wave[0])
        axs[i, 2].set_title(f'Denoised Wave (SNR={snrs[i]})')
        axs[i, 2].set_ylim(min_wave, max_wave)
        remove_frame(axs[i, 2])  # Entferne den Rahmen
        if i < 2:  # Nur im untersten Plot eine x-Achse anzeigen
            axs[i, 2].set_xticks([])
        else:# Für den untersten Plot die Linie und die Beschriftung anzeigen
            set_bottom_line(axs[i, 2])
            axs[i, 2].set_xlabel('Time')
        axs[i, 2].set_yticks([])

    # Beschriftung der x-Achse (nur für den unteren Plot)
    axs[2, 1].set_xlabel('Time')  # X-Achse für zweite Spalte (unten)
    axs[2, 2].set_xlabel('Time')  # X-Achse für dritte Spalte (unten)

    # Layout anpassen
    plt.tight_layout()
    #plt.show()
    return fig

def generate_das_plot3(clean_das, all_noisy_waves, all_denoised_waves, all_semblance, snr_indices, vmin, vmax):
    # Übertrage die Daten auf die CPU
    clean_das_cpu = clean_das.to('cpu').detach().numpy()
    noisy_waves_cpu = all_noisy_waves.to('cpu').detach().numpy()
    denoised_waves_cpu = all_denoised_waves.to('cpu').detach().numpy()
    semblance_cpu = all_semblance.to('cpu').detach().numpy()

    # Berechne das globale Minimum und Maximum der Daten für eine einheitliche Farbgebung
    #vmin = -1e-08 #clean_das_cpu.min() = -0.00000013847445 or -1.3847445e-07
    #vmax = 1e-08 #clean_das_cpu.max() = 1.0335354e-07
    #vmin = np.percentile(clean_das_cpu, 9)
    #vmax = np.percentile(clean_das_cpu, 91)

    # Erstelle das Grid (3 Spalten und 8 Zeilen)
    fig, axs = plt.subplots(3, 4, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 1, 1, 1.1], 'height_ratios': [1, 1, 1]})

    # Spalte 1: Clean DAS (mittig, über vier Zeilen)
    axs[1, 0].imshow(clean_das_cpu, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis', interpolation="antialiased", rasterized=True)
    axs[1, 0].set_title('Clean DAS')
    #remove_frame(axs[1, 0])  # Entferne den Rahmen
    #set_bottom_line(axs[1, 0])  # Zeige nur die untere Linie
    #axs[1, 0].yaxis.set_visible(False)  # Deaktiviere y-Achsen für das mittige Diagramm
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Channel Index')
    # Leere Felder in der ersten Spalte
    axs[0, 0].set_axis_off()  # Oberes Feld leer
    axs[2, 0].set_axis_off()  # Unteres Feld leer
    
    # Spalte 2: Noisy DAS (SNR 0.1 und SNR 10 in Zeilen)
    for i, snr_idx in enumerate(snr_indices):
        axs[i, 1].imshow(noisy_waves_cpu[i], aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis', interpolation="antialiased", rasterized=True)
        axs[i, 1].set_title(f'Input DAS (SNR={snr_idx})')
        if i == 2:
            axs[i, 1].set_xlabel('Time')
            axs[i, 1].set_ylabel('Channel Index')

    # Spalte 3: Denoised DAS (SNR 0.1 und SNR 10 in Zeilen)
    for i, snr_idx in enumerate(snr_indices):
        axs[i, 2].imshow(denoised_waves_cpu[i], aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis', interpolation="antialiased", rasterized=True)
        axs[i, 2].set_title(f'Denoised DAS (SNR={snr_idx})')
        if i == 2:# Für den untersten Plot die Linie und die Beschriftung anzeigen
            axs[i, 2].set_xlabel('Time')
            axs[i, 2].set_ylabel('Channel Index')

    #Spalte 4: Semblance (SNR 0.1 und SNR 10 in Zeilen)
    for i, snr_idx in enumerate(snr_indices):
        im = axs[i, 3].imshow(semblance_cpu[i],  origin='lower', interpolation='nearest', cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=axs[i, 3], orientation='vertical')
        axs[i, 3].set_title(f'Semblance (SNR={snr_idx})')
        if i == 2:# Für den untersten Plot die Linie und die Beschriftung anzeigen
            axs[i, 3].set_xlabel('Time')
            axs[i, 3].set_ylabel('Channel Index')

    # Lege das Layout fest und zeige den Plot
    plt.tight_layout()
    #plt.show()
    return fig

def generate_real_das_plot(clean_das, all_denoised_das, semblance_das, channel_idx_1, channel_idx_2, vmin, vmax, min_wave, max_wave):
        """Args:
        clean_das_original: Clean DAS sample (torch.Tensor)
        real_denoised: Denoised DAS sample (torch.Tensor)
        channel_idx_1: first highlighted chanel (int)
        channel_idx_2: secound highlighted chanel (int)
        vmin: Min value for DAS plot (float)
        vmax: Max value for DAS plot (float)
        min_wave: Min value for the wave plot (float)
        max_wave: Max value for the wave plot (float)
        """
        # Plot clean_das_original and real_denoised as imshows in one row
        fig, axs = plt.subplots(3, 4, figsize=(15, 12), 
                        gridspec_kw={'width_ratios': [1, 1, 1, 0.05], 'height_ratios': [3, 1, 1]})
        axs[0, 0].imshow(clean_das, origin='lower', interpolation='nearest', cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
        axs[0, 0].set_title('Real DAS Sample')
        axs[0, 0].set_ylabel('Channel Index')
        axs[0, 0].axhline(y=channel_idx_1, color='blue', linestyle='--', linewidth=2, label=f'Channel {channel_idx_1}')
        axs[0, 0].axhline(y=channel_idx_2, color='green', linestyle='--', linewidth=2, label=f'Channel {channel_idx_2}')

        axs[0, 1].imshow(all_denoised_das, origin='lower', interpolation='nearest', cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
        axs[0, 1].set_title('Denoised')
        axs[0, 1].axhline(y=channel_idx_1, color='blue', linestyle='--', linewidth=2, label=f'Channel {channel_idx_1}')
        axs[0, 1].axhline(y=channel_idx_2, color='green', linestyle='--', linewidth=2, label=f'Channel {channel_idx_2}')

        axs[0, 2].imshow(semblance_das, origin='lower', interpolation='nearest', cmap='viridis', aspect='auto')
        axs[0, 2].set_title('Semblance')
        axs[0, 1].axhline(y=channel_idx_1, color='blue', linestyle='--', linewidth=2, label=f'Channel {channel_idx_1}')
        axs[0, 1].axhline(y=channel_idx_2, color='green', linestyle='--', linewidth=2, label=f'Channel {channel_idx_2}')

        #color bar for semblance
        im = axs[0, 2].imshow(semblance_das[0], origin='lower', interpolation='nearest', cmap='viridis', aspect='auto')
        fig.colorbar(im, cax=axs[0, 3])
        remove_frame(axs[1, 3])
        axs[1, 3].set_yticks([])
        axs[1, 3].set_xticks([])
        remove_frame(axs[2, 3]) 
        axs[2, 3].set_yticks([])
        axs[2, 3].set_xticks([])

        #Wellenform ploten
        axs[1, 0].plot(clean_das[channel_idx_1], color='blue')
        axs[1, 0].set_ylim(min_wave, max_wave)
        axs[1, 0].set_ylabel(f'Kanal {channel_idx_1}')
        remove_frame(axs[1, 0])  # Entferne den Rahmen
        set_bottom_line(axs[1, 0], True)  # Zeige nur die untere Linie
        axs[1, 0].set_yticks([])


        axs[1, 1].plot(all_denoised_das[channel_idx_1], color='blue')
        axs[1, 1].set_ylim(min_wave, max_wave)
        remove_frame(axs[1, 1])  # Entferne den Rahmen
        set_bottom_line(axs[1, 1], True)  # Zeige nur die untere Linie
        axs[1, 1].set_yticks([])

        axs[1, 2].plot(semblance_das[channel_idx_1], color='blue')
        axs[1, 2].set_ylim(min_wave, max_wave)
        remove_frame(axs[1, 2])  # Entferne den Rahmen
        set_bottom_line(axs[1, 2])  # Zeige nur die untere Linie
        axs[1, 2].set_yticks([])

        axs[2, 0].plot(clean_das[channel_idx_2], color='green')
        axs[2, 0].set_ylim(min_wave, max_wave)
        axs[2, 0].set_ylabel(f'Kanal {channel_idx_2}')
        axs[2, 0].set_xlabel('Time')
        remove_frame(axs[2, 0])  # Entferne den Rahmen
        set_bottom_line(axs[2, 0], True)  # Zeige nur die untere Linie
        axs[2, 0].set_yticks([])

        axs[2, 1].plot(all_denoised_das[channel_idx_2], color='green')
        axs[2, 1].set_ylim(min_wave, max_wave)
        axs[2, 1].set_xlabel('Time')
        remove_frame(axs[2, 1])  # Entferne den Rahmen
        set_bottom_line(axs[2, 1], True)  # Zeige nur die untere Linie
        axs[2, 1].set_yticks([])

        axs[2, 2].plot(semblance_das[channel_idx_2], color='green')
        axs[2, 2].set_ylim(min_wave, max_wave)
        axs[2, 2].set_xlabel('Time')
        remove_frame(axs[2, 2])  # Entferne den Rahmen
        set_bottom_line(axs[2, 2])  # Zeige nur die untere Linie
        axs[2, 2].set_yticks([])

        plt.tight_layout()
        return fig
