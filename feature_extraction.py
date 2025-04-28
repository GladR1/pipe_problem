import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from numpy.fft import fft
import pandas as pd

def extract_stat_features(signal):

    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'range': np.max(signal) - np.min(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal),
        'energy': np.sum(signal ** 2),
    }
    return features

def extract_peaks_features(signal):

    peaks, _ = find_peaks(signal)
    valleys, _ = find_peaks(-signal)
    return {
        'n_peaks': len(peaks),
        'n_valleys': len(valleys),
    }

def extract_fft_features(signal, n_components=5):
    fft_coeffs = np.abs(fft(signal))

    fft_features = {
        f'fft_{i}': fft_coeffs[i] for i in range(1, n_components + 1)
    }
    return fft_features

def extract_features_for_id(group):

    features = {'id': group['id'].iloc[0]}
    
    for ch in ['ch0', 'ch1', 'ch2']:
        signal = group[ch].values
        
        stat_feats = extract_stat_features(signal)
        stat_feats = {f'{ch}_{k}': v for k, v in stat_feats.items()}
        
        peaks_feats = extract_peaks_features(signal)
        peaks_feats = {f'{ch}_{k}': v for k, v in peaks_feats.items()}
        
        fft_feats = extract_fft_features(signal)
        fft_feats = {f'{ch}_{k}': v for k, v in fft_feats.items()}
        
        features.update(stat_feats)
        features.update(peaks_feats)
        features.update(fft_feats)
    
    return features

def build_features(data):

    features_list = []
    for _, group in data.groupby('id'):
        feats = extract_features_for_id(group)
        features_list.append(feats)
    return pd.DataFrame(features_list)