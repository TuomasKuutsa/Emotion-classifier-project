import numpy as np
import librosa as lr
import librosa.feature as lrf
from os import getpid


def feature_extractor(path, n_masks, for_sklearn, n_mels, n_mfccs, only_mfccs):
    
    print(f'processing with id: {getpid()} file: {path}')
    
    audio, sr = lr.load(path, sr=None, mono=True, offset=0.7, duration=2.2)
        
    log_mels = []
    
    log_mels.append(lr.power_to_db(lrf.melspectrogram(audio, sr=sr, n_fft=512, n_mels=n_mels, fmax=8000)))

    log_mels = log_mels + augmentation(audio, sr, n_masks, n_mels)
    
    mfccs  = [lrf.mfcc(S=s, n_mfcc=n_mfccs) for s in log_mels]
    
    target_list = []
    label = int(path.split('-')[6])
    target_list.extend([label]*(n_masks+1))

    if for_sklearn:
        
        sample = take_means(log_mels, mfccs)

        gender = int((path.split('-')[-1].split('.')[0]))
        if gender % 2 == 0:
            sample = [np.append(s, 1) for s in sample]
        else:
            sample = [np.append(s, 0) for s in sample]

        intensity = int(path.split('-')[7])
        if intensity == 1:
            sample = [np.append(s, 1) for s in sample]
        else:
            sample = [np.append(s, 0) for s in sample]

        statement = int(path.split('-')[8])
        if statement == 1:
            sample = [np.append(s, 1) for s in sample]
        else:
            sample = [np.append(s, 0) for s in sample]

        return sample, target_list
    
    else:
        return combine(log_mels, mfccs, n_mels, only_mfccs), target_list
    
def augmentation (audio, sr, n_masks, n_mels):
    
    copy_list = []
    
    for _ in range(n_masks):
        
        copy = audio.copy()
        
        copy = copy + np.random.uniform(-0.002, 0.002, size=len(copy))
        
        copy = lr.effects.pitch_shift(copy, sr, n_steps=np.random.randint(-12, 13), bins_per_octave=24)
        
        copy = lr.power_to_db(lrf.melspectrogram(audio, sr=sr, n_fft=512, n_mels=n_mels, fmax=8000))
        
        copy = masker(copy, n_mels)
        
        copy_list.append(copy)
        
    return copy_list

def masker(mel, n_mels):
        
    copy = mel.copy()

    f_mask_start1 = np.random.randint(2, n_mels/6)
    f_mask_end1 =  f_mask_start1 + np.random.randint(3, 5)

    f_mask_start2 = np.random.randint(n_mels/6, n_mels/2)
    f_mask_end2 =  f_mask_start2 + np.random.randint(4, 8)

    t_mask_start = np.random.randint(15, 50)
    t_mask_end = t_mask_start + np.random.randint(3, 5)

    copy[f_mask_start1:f_mask_end1] = np.random.randint(-90, -50)
    copy[f_mask_start2:f_mask_end2] = np.random.randint(-90, -50)
    copy[:,t_mask_start:t_mask_end] = np.random.randint(-90, -50)
        
    return copy

def take_means(log_mels, mfccs):

    samples = []
    for mel, mfcc in zip(log_mels, mfccs):

        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        sample = np.concatenate((mel_mean, mel_std, mfcc_mean, mfcc_std), axis=None)
        
        samples.append(sample)

    return samples

def combine(mels, mfccs, n_mels, only_mfccs):
    
    samples = []
    for mel, mfcc in zip(mels, mfccs):
        
        mel = lr.util.normalize(mel, axis=1)
        mfcc = lr.util.normalize(mfcc, axis=1)
        
        sample = np.concatenate([mel, mfcc])
        sample = sample.reshape(sample.shape[0], sample.shape[1], 1)
        if only_mfccs:
            sample = sample[n_mels:]
            samples.append(sample)
        else:
            samples.append(sample)
    
    return samples