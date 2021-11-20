import pandas as pd
import numpy as np
import librosa as lr
import librosa.feature as lrf
from os import getpid


def feature_extractor(file_path, n_masks, returns_data):
    
    print(f'processing with id: {getpid()} file: {file_path}')
    
    audio, _ = lr.load(file_path, mono=True, offset=0.7, duration=2.2)
    
    log_mel_spectrograms = []
    
    log_mel_spectrograms.append(lr.power_to_db(lrf.melspectrogram(audio, n_fft=512)))
    log_mel_spectrograms = log_mel_spectrograms + masker(log_mel_spectrograms[0], n_masks)
    
    mfccs, delta_mfccs, delta2_mfccs = extract_mfccs_and_derivatives(log_mel_spectrograms)

    chromas = [lrf.chroma_stft(lrf.inverse.mel_to_audio(lr.db_to_power(spectrogram), n_fft=512), n_fft=512, hop_length=128) for spectrogram in log_mel_spectrograms]
    
    sample = flatten_features(log_mel_spectrograms, mfccs, delta_mfccs, delta2_mfccs, chromas)
    
    if returns_data:
        return sample
    return log_mel_spectrograms, mfccs, delta_mfccs, delta2_mfccs, chromas

def masker(log_mel_spectrogram, n_masks):
    
    copy_list = []

    for _ in range(n_masks):
        
        spectrogram_copy = log_mel_spectrogram.copy()
        frequency_mask_start = np.random.randint(15, 50)
        frequency_mask_end =  frequency_mask_start + np.random.randint(4, 9)

        time_mask_start = np.random.randint(35, 115)
        time_mask_end =time_mask_start + np.random.randint(4, 9)

        spectrogram_copy[frequency_mask_start:frequency_mask_end] = np.random.randint(-90, -40)
        spectrogram_copy[:,time_mask_start:time_mask_end] = np.random.randint(-90, -40)
        
        copy_list.append(spectrogram_copy)
    
    return copy_list


def extract_mfccs_and_derivatives(log_mel_spectrograms):
    
    mfccs = []
    delta_mfccs = []
    delta2_mfccs = []
    
    for spectrogram in log_mel_spectrograms:
        mfcc = lrf.mfcc(S=spectrogram, n_mfcc=24)
        mfccs.append(mfcc)
        delta_mfccs.append(lrf.delta(mfcc))
        delta2_mfccs.append(lrf.delta(mfcc, order=2))
        
    return mfccs, delta_mfccs, delta2_mfccs

    
def flatten_features(log_mel_spectrograms, mfccs, delta_mfccs, delta2_mfccs, chromas):
    
    flattened_features = []
    for s, m, dm, d2m, ch in zip(log_mel_spectrograms, mfccs, delta_mfccs, delta2_mfccs, chromas):
        flattened_features.append(np.concatenate([s, m, dm, d2m, ch]).flatten())
        
    return flattened_features