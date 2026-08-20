import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from pathlib import Path
import os
from mpl_toolkits.mplot3d import Axes3D
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

from scipy import signal, stats
from scipy.signal import butter, lfilter

from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


# CONFIGURATION & CONSTANTS
TARGET_CHANNELS = [
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'Fpz', 'Fp1', 'Fp2', 'Cz', 'CPz',
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'
]

COORDS = {
    'FP1': (-0.31, 0.95, 0), 'FPz': (0.0, 1.0, 0), 'FP2': (0.31, 0.95, 0),
    'AF7': (-0.59, 0.81, 0), 'AF8': (0.59, 0.81, 0), 'F7': (-0.81, 0.59, 0),
    'F5': (-0.71, 0.67, 0.23), 'F3': (-0.57, 0.7, 0.44), 'F1': (-0.3, 0.74, 0.6),
    'Fz': (0.0, 0.71, 0.71), 'F2': (0.3, 0.74, 0.6), 'F4': (0.57, 0.7, 0.44),
    'F6': (0.71, 0.67, 0.23), 'F8': (0.81, 0.59, 0), 'FT7': (-0.95, 0.31, 0),
    'FT8': (0.95, 0.31, 0), 'FC5': (-0.88, 0.36, 0.33), 'FC3': (-0.69, 0.38, 0.62),
    'FC1': (-0.37, 0.41, 0.83), 'FCz': (0.0, 0.38, 0.92), 'FC2': (0.37, 0.41, 0.83),
    'FC4': (0.69, 0.38, 0.62), 'FC6': (0.88, 0.36, 0.33), 'T7': (-1.0, 0.0, 0),
    'T8': (1.0, 0.0, 0), 'C5': (-0.92, 0.0, 0.38), 'C3': (-0.71, 0.0, 0.71),
    'C1': (-0.38, 0.0, 0.92), 'Cz': (0.0, 0.0, 1.0), 'C2': (0.38, 0.0, 0.92),
    'C4': (0.71, 0.0, 0.71), 'C6': (0.92, 0.0, 0.38), 'TP7': (-0.95, -0.31, 0),
    'CP5': (-0.88, -0.36, 0.33), 'CP3': (-0.69, -0.38, 0.62), 'CP1': (-0.37, -0.41, 0.83),
    'CPz': (0.0, -0.38, 0.92), 'CP2': (0.37, -0.41, 0.83), 'CP4': (0.69, -0.38, 0.62),
    'CP6': (0.88, -0.36, 0.33), 'TP8': (0.95, -0.31, 0), 'P7': (-0.81, -0.59, 0),
    'P5': (-0.71, -0.67, 0.23), 'P3': (-0.57, -0.7, 0.44), 'P1': (-0.3, -0.74, 0.6),
    'Pz': (0.0, -0.71, 0.71), 'P2': (0.3, -0.74, 0.6), 'P4': (0.57, -0.7, 0.44),
    'P6': (0.71, -0.67, 0.23), 'P8': (0.81, -0.59, 0), 'PO7': (-0.59, -0.81, 0.23),
    'PO3': (-0.46, -0.86, 0.38), 'POz': (0.0, -0.92, 0.23), 'PO4': (0.46, -0.86, 0.38),
    'PO8': (0.59, -0.81, 0.23), 'O1': (-0.31, -0.95, 0), 'Oz': (0.0, -1.0, 0),
    'O2': (0.31, -0.95, 0), 'AF4': (0.46, 0.86, 0.23), 'AF5': (-0.46, 0.86, 0.23),
    'AF9': (-0.84, 0.49, -0.24), 'AF10': (0.84, 0.49, -0.24)
}

SFREQ = 512 
WINLEN = 512
OVERLAP = 256
FREQUENCY_BANDS = [(3.5, 8), (8, 12), (12, 30), (30, 45)]
BAND_NAMES = ['theta', 'alpha', 'beta', 'gamma']


class FeatureExtraction:
    
    def det_feat(epochs, highpass, lowpass, labels, outdir, feature, typ):
        chnames = epochs.ch_names

        if feature == 'Chaos':
            print(f'[INFO] Creating {typ} chaos features...')
            Chaos.fanalysis(epochs, labels, chnames, outdir)
            print(f'[INFO] Saved {typ} Chaos features.')


        elif feature == 'Linear':
            print(f'[INFO] Creating {typ} linear features...')
            Linear.fanalysis(epochs, labels, chnames, outdir, lowpass, highpass)
            print(f'[INFO] Saved {typ} linear features.')




class Linear:

    def bandpassfilter(data, lowcut, highcut, sfreq, order= 5):
        nyq = 0.5 * sfreq
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data, axis=-1)


    def slintegrals(sig, winlen, step, dt):
        n_samples = sig.shape[0]
        n_windows = max(0, (n_samples - winlen) // step + 1)
        
        if n_windows == 0:
            return np.array([])

        out = np.empty(n_windows, dtype=np.float32)
        for i in range(n_windows):
            start = i * step
            end = start + winlen
            window = sig[start:end]

            out[i] = np.trapz(np.abs(window), dx=dt)
            
        return out

    def wintegrals(epochs, labels, chnames, outdir):

        data = epochs.get_data()
        sfreq = SFREQ
        
        winlen = WINLEN
        overlap = OVERLAP
        step = winlen - overlap
        dt = 1.0 / sfreq
        
        n_epochs, n_channels, n_times = data.shape
        n_bands = len(FREQUENCY_BANDS)
        n_windows = (n_times - winlen) // step + 1
        
        if n_windows <= 0:
            raise ValueError(f"[ERROR] Signal too short for window length {winlen} with sampling rate {sfreq}")
        
        print(f"[INFO] Processing {n_epochs} epochs, {n_channels} channels, {n_bands} bands...")


        csv_path = os.path.join(outdir, 'Wintegrals.csv')
        fieldnames = ['epoch', 'label', 'frequency_band', 'channel', 'window_idx', 'integral']
        
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            

            for ep in range(n_epochs):
                current_label = labels[ep]
                
                for bidx, (low, high) in enumerate(FREQUENCY_BANDS):
                    band_name = BAND_NAMES[bidx]
                    
                    filtered_band = np.empty_like(data[ep:ep+1]) 
                    for ch in range(n_channels):
                        filtered_band[0, ch, :] = Linear.bandpassfilter(
                            data[ep, ch, :], low, high, sfreq
                        )
                    

                    for ch in range(n_channels):
                        ch_signal = filtered_band[0, ch, :]
                        
                        int_vals = Linear.slintegrals(ch_signal, winlen, step, dt)
                        
                        ch_name = chnames[ch]
                        
                        for w_idx, val in enumerate(int_vals):
                            writer.writerow({
                                'epoch': ep,
                                'label': int(current_label),
                                'frequency_band': band_name,
                                'channel': ch_name,
                                'window_idx': w_idx,
                                'integral': val
                            })
        
        print(f"[INFO] Saved integrals to: {csv_path}")

    
    def fanalysis(epochs, labels, chnames, outdir, lowpass, highpass):

        
        print(f"[INFO] Starting Linear Feature Extraction...")
        print(f"  Input: {epochs.get_data().shape}, Bands: {FREQUENCY_BANDS}")

        Linear.wintegrals(epochs, labels, chnames, outdir)
        
        print(f"[INFO] Linear feature extraction completed. Output saved to {outdir}/Wintegrals.csv")

class Chaos:
    def Hcore(signal, kmax=5):

        N = len(signal)
        if N < kmax: return 0.0
        lengths = np.zeros(kmax)
        for k in range(1, kmax + 1):
            Lk = 0.0
            count = 0
            for m in range(k):
                subseq = signal[m::k]
                if len(subseq) < 2: continue
                diffs = np.abs(np.diff(subseq))
                sumdiff = np.sum(diffs)
                Lm = (sumdiff * (N - 1)) / ((N - m) * k)
                Lk += Lm
                count += 1
            if count > 0: lengths[k-1] = Lk / count
            else: lengths[k-1] = 0.0
        
        x = np.log(1.0 / np.arange(1, kmax + 1))
        y = np.log(lengths + 1e-10)

        slope, _ = np.polyfit(x, y, 1)

        return slope


    def lyapu(X, min_separation=10, max_time=50):

        L, emb_dim = X.shape
        if L < min_separation + max_time: return 0.0

        distances = cdist(X, X, metric='euclidean')
        np.fill_diagonal(distances, np.inf)
        
        time_indices = np.arange(L)
        time_diff = np.abs(time_indices[:, None] - time_indices[None, :])
        distances[time_diff < min_separation] = np.inf
        
        nearest_neighbors = np.argmin(distances, axis=1)
        min_dist = np.min(distances, axis=1)
        valid_mask = min_dist < np.inf
        
        if not np.any(valid_mask): return 0.0

        log_distances = []
        time_steps = np.arange(max_time)
        
        for k in time_steps:
            div_sum = 0.0
            count = 0
            for i in range(L):
                if not valid_mask[i]: continue
                j = nearest_neighbors[i]
                if i + k >= L or j + k >= L: continue
                
                dist_k = np.linalg.norm(X[i+k] - X[j+k])
                if dist_k > 0:
                    div_sum += np.log(dist_k)
                    count += 1
            
            if count > 0: log_distances.append(div_sum / count)
            else: log_distances.append(np.nan)
        
        log_distances = np.array(log_distances)
        valid_indices = ~np.isnan(log_distances)
        if np.sum(valid_indices) < 2: return 0.0
        
        fit_range = min(15, np.sum(valid_indices))
        x_fit = time_steps[:fit_range]
        y_fit = log_distances[:fit_range]
        
        mask = ~np.isnan(y_fit)
        x_fit = x_fit[mask]
        y_fit = y_fit[mask]
        
        if len(x_fit) < 2: return 0.0
        try: return np.polyfit(x_fit, y_fit, 1)[0]
        except: return 0.0


    def psplots(epochs, labels, chnames, base_path):

        embed_dir = Path(base_path) / "delay_embeddings"
        embed_dir.mkdir(parents=True, exist_ok=True)
        
        ntrials, nchannels, ntimes = epochs.shape
        unique_classes = np.unique(labels)
        
        delay = 5
        dim = 3
        L = ntimes - (dim - 1) * delay
        
        if L <= 0:
            print(f"[ERROR] Signal too short for embedding.")
            return

        print(f"[INFO] Generating 3D Delay Embeddings...")

        for cls in unique_classes:
            mask = (labels == cls)
            clstrials = epochs[mask]
            class_dir = embed_dir / f"class_{cls}"
            class_dir.mkdir(exist_ok=True)
            
            for chidx in range(nchannels):
                chname = chnames[chidx]
                
                if clstrials.shape[0] > 10:
                    signals_to_plot = [np.mean(clstrials[:, chidx, :], axis=0)]
                    titles = [f"Avg Class {cls} - {chname}"]
                else:
                    signals_to_plot = [clstrials[i, chidx, :] for i in range(clstrials.shape[0])]
                    titles = [f"Trial {i} - Class {cls} - {chname}" for i in range(clstrials.shape[0])]
                
                for sig, title in zip(signals_to_plot, titles):
                    N = len(sig)
                    if N < (dim - 1) * delay: continue
                    
                    X = np.zeros((L, dim))
                    for d in range(dim):
                        start_idx = d * delay
                        end_idx = N - (dim - 1 - d) * delay
                        segment = sig[start_idx:end_idx]
                        if len(segment) != L: continue
                        X[:, d] = segment
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    t = np.arange(L)
                    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='plasma', s=2, alpha=0.7)
                    ax.set_title(title)
                    ax.set_xlabel(f"X (t)")
                    ax.set_ylabel(f"Y (t+{delay})")
                    ax.set_zlabel(f"Z (t+{2*delay})")
                    
                    safe_name = title.replace(" ", "_").replace("/", "_").replace(":", "")
                    filepath = class_dir / f"{safe_name}.png"
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close()
        
        print(f"[INFO] Delay Embeddings saved to: {embed_dir}")


    def topo(epochs, labels, chnames, outdir, hfd_stats, custom_montage=COORDS):
        topodir = Path(outdir) / "topographies"
        os.makedirs(topodir, exist_ok=True)

        unique_classes = np.unique(labels)
        valid_channels = [ch for ch in chnames if ch in custom_montage]
        
        if not valid_channels:
            print("[ERROR] No channels matched between data and custom montage.")
            return
        
        dig_montage = mne.channels.make_dig_montage(
            ch_pos={ch: custom_montage[ch] for ch in valid_channels},
            coord_frame='head'
        )
        info = mne.create_info(ch_names=valid_channels, sfreq=1, ch_types='eeg')
        info.set_montage(dig_montage, match_case=False, match_alias=False)
        
        print(f"[INFO] Prepared Info object with {len(info.ch_names)} channels for topography.")

        for cls in unique_classes:
            if cls not in hfd_stats: continue
            hfd_means = hfd_stats[cls]['hfd_mean']
            
            aligned_data = []
            for ch in valid_channels:
                if ch in chnames:
                    idx = chnames.index(ch)
                    aligned_data.append(hfd_means[idx])
            
            if len(aligned_data) != len(valid_channels): continue

            fig, ax = plt.subplots(figsize=(8, 8))
            data_min, data_max = np.min(aligned_data), np.max(aligned_data)
            margin = (data_max - data_min) * 0.02
            
            im, _ = mne.viz.plot_topomap(aligned_data, info, axes=ax, cmap='viridis', contours=6, sphere='auto', show=False)
            im.set_clim(data_min - margin, data_max + margin)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Mean HFD')
            ax.set_title(f'Topography: Class {cls}')
            
            filename = f"topo_class_{cls}.png"
            plt.savefig(topodir / filename, dpi=150, bbox_inches='tight')
            plt.close()


    def fanalysis(epochs, labels, chnames, outdir, kmax=5, delay=5, dim=3):
        
        epochs_array= epochs.get_data()
        ntrials, nchannels, ntimes = epochs_array.shape
        unique_classes = np.unique(labels)
        
        topodir = Path(outdir) / "topographies"
        os.makedirs(topodir, exist_ok=True)
        embedir = Path(outdir) / "Delay Embeddings"
        os.makedirs(embedir,exist_ok=True)

        lyap_rows = [] 
        full_classstats = {}
        L_embed = ntimes - (dim - 1) * delay
        
        if L_embed <= 0:
            print("[ERROR] Signal too short for embedding.")
            return

        print(f"[INFO] Processing {ntrials} trials, {nchannels} channels...")
        print(f"[INFO] Embedding Params: delay={delay}, dim={dim}, L={L_embed}")

        for cls in unique_classes:
            mask = (labels == cls)
            clstrials = epochs_array[mask]
            n_clstrials = clstrials.shape[0]
            if n_clstrials == 0: continue
            
            hmean = np.zeros(nchannels)
            hstd = np.zeros(nchannels)
            lmean = np.zeros(nchannels)
            lstd = np.zeros(nchannels)
            
            # New stats for Phase Space
            vol_mean = np.zeros(nchannels)
            vol_std = np.zeros(nchannels)
            cent_x_mean = np.zeros(nchannels)
            cent_y_mean = np.zeros(nchannels)
            cent_z_mean = np.zeros(nchannels)
            
            print(f"[INFO] Processing Class {cls} ({n_clstrials} trials)...")

            for chidx in range(nchannels):
                chname = chnames[chidx]
                chdata = clstrials[:, chidx, :]
                
                h_vals = []
                l_vals = []
                vol_vals = []
                cent_x_vals = []
                cent_y_vals = []
                cent_z_vals = []
                
                for t in range(n_clstrials):
                    ts = chdata[t]
                    ts = ts - np.mean(ts)
                    
                    h_val = Chaos.Hcore(ts, kmax=kmax)
                    h_vals.append(h_val)
                    
                    if L_embed > 0:
                        X = np.zeros((L_embed, dim))
                        for d in range(dim):
                            start_idx = d * delay
                            end_idx = len(ts) - (dim - 1 - d) * delay
                            X[:, d] = ts[start_idx:end_idx]
                        
                        l_val = Chaos.lyapu(X)
                        l_vals.append(l_val)
                        
                        centroid = np.mean(X, axis=0)
                        cent_x_vals.append(centroid[0])
                        cent_y_vals.append(centroid[1])
                        cent_z_vals.append(centroid[2])
                        
                        try:
                            if L_embed > dim:
                                hull = ConvexHull(X)
                                vol = hull.volume 
                            else:
                                vol = 0.0
                        except:
                            vol = 0.0
                        vol_vals.append(vol)
                        
   
                        lyap_rows.append({
                            'Trial': np.where(mask)[0][t],
                            'Class': cls,
                            'Channel': chname,
                            'Lyapunov_Exponent': l_val,
                            'HFD': h_val, 
                            'Embedding_Delay': delay,
                            'Embedding_Dim': dim,
                            'Embedding_Length': L_embed,
                            'Phase_Centroid_X': centroid[0],
                            'Phase_Centroid_Y': centroid[1],
                            'Phase_Centroid_Z': centroid[2],
                            'Phase_Volume': vol
                        })
                    else:
                        l_vals.append(0.0)
                        vol_vals.append(0.0)
                        cent_x_vals.append(0.0)
                        cent_y_vals.append(0.0)
                        cent_z_vals.append(0.0)
                        lyap_rows.append({
                            'Trial': np.where(mask)[0][t],
                            'Class': cls,
                            'Channel': chname,
                            'Lyapunov_Exponent': 0.0,
                            'HFD': 0.0,
                            'Embedding_Delay': delay,
                            'Embedding_Dim': dim,
                            'Embedding_Length': 0,
                            'Phase_Centroid_X': 0.0,
                            'Phase_Centroid_Y': 0.0,
                            'Phase_Centroid_Z': 0.0,
                            'Phase_Volume': 0.0
                        })
                

                hmean[chidx] = np.mean(h_vals)
                hstd[chidx] = np.std(h_vals)
                lmean[chidx] = np.mean(l_vals)
                lstd[chidx] = np.std(l_vals)
                
                vol_mean[chidx] = np.mean(vol_vals)
                vol_std[chidx] = np.std(vol_vals)
                cent_x_mean[chidx] = np.mean(cent_x_vals)
                cent_y_mean[chidx] = np.mean(cent_y_vals)
                cent_z_mean[chidx] = np.mean(cent_z_vals)
            
            full_classstats[cls] = {
                'hfd_mean': hmean, 'hfd_std': hstd,
                'lyap_mean': lmean, 'lyap_std': lstd,
                'vol_mean': vol_mean, 'vol_std': vol_std,
                'cent_x_mean': cent_x_mean, 'cent_y_mean': cent_y_mean, 'cent_z_mean': cent_z_mean
            }

        df_lyap = pd.DataFrame(lyap_rows)
        df_lyap.to_csv(os.path.join(outdir, 'Raw_Lyapunov_Full.csv'), index=False)
        print(f"[INFO] Saved Lyapunov CSV with {len(lyap_rows)} rows (includes coordinates & params).")


        print("\n[INFO] Generating HFD Topographies...")
        Chaos.topo(epochs_array, labels, chnames, outdir, full_classstats)


        print("\n[INFO] Generating Delay Embeddings ...")
        matched_indices = [chnames.index(n) for n in TARGET_CHANNELS if n in chnames]
        if matched_indices:
            filtered_epochs = epochs_array[:, matched_indices, :]
            filtered_chnames = [chnames[i] for i in matched_indices]
            Chaos.psplots(filtered_epochs, labels, filtered_chnames, outdir)
        else:
            Chaos.psplots(epochs_array, labels, chnames, outdir)

        print("\n[INFO] FULL CHAOS ANALYSIS COMPLETE!")
        return filtered_epochs if matched_indices else epochs_array, full_classstats