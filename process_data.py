import numpy as np
import librosa as lr
import librosa.feature as lrf
from os import getpid

from multiprocessing import Pool


def process(path, for_sklearn, n_augmentations, n_mels, full_mels, n_mfccs, fmax):
    
    print(f'processing with id: {getpid()} file: {path}')
    
    audio, sr = lr.load(path, sr=None, mono=True, offset=0.7, duration=2.2)
        
    mels = []
    
    mels.append(lr.power_to_db(lrf.melspectrogram(audio, sr=sr, n_fft=512, n_mels=n_mels, fmax=fmax)))

    mels = mels + augmentation(audio, sr, n_augmentations, n_mels, fmax)
    
    mfccs  = [lrf.mfcc(S=s, n_mfcc=n_mfccs) for s in mels]

    gender = 0
    if int((path.split('-')[-1].split('.')[0])) % 2 == 0:
            gender = 1
    intensity = int(path.split('-')[7])-1
    statement = int(path.split('-')[8])-1
    
    target_list = []
    label = int(path.split('-')[6])
    target_list.extend([label]*(n_augmentations+1))

    if for_sklearn:
        
        sample = flatten(mels, mfccs)
        sample = [np.append(s, gender) for s in sample]
        sample = [np.append(s, intensity) for s in sample]
        sample = [np.append(s, statement) for s in sample]

        return sample, target_list
    
    else:

        mels, mfccs = reshape(mels, mfccs, full_mels)
        meta = [np.asarray([gender, intensity, statement], dtype=np.float32) for _ in range(n_augmentations+1)]       

        return mfccs, mels, meta, target_list


   
def augmentation (audio, sr, n_augmentations, n_mels, fmax):
    
    copy_list = []
    
    for _ in range(n_augmentations):

        copy = audio.copy()
        
        copy = copy + np.random.normal(0, 0.0001, size=len(audio))
        
        # pitch_shift = lr.effects.pitch_shift(copy, sr, n_steps=np.random.randint(-2, 3), bins_per_octave=24)
        
        mel = lr.power_to_db(lrf.melspectrogram(copy, sr=sr, n_fft=512, n_mels=n_mels, fmax=fmax))
        
        mel = masker(mel, n_mels)
        
        copy_list.append(mel)
        
    return copy_list

def masker(mel, n_mels):

    f_mask_start1 = np.random.randint(0, n_mels/8)
    f_mask_end1 =  f_mask_start1 + np.random.randint(3, 5)

    f_mask_start2 = np.random.randint(n_mels/8, n_mels/2)
    f_mask_end2 =  f_mask_start2 + np.random.randint(3, 5)

    t_mask_start = np.random.randint(15, 50)
    t_mask_end = t_mask_start + np.random.randint(3, 5)

    mel[f_mask_start1:f_mask_end1] = np.random.randint(-90, -50)
    mel[f_mask_start2:f_mask_end2] = np.random.randint(-90, -50)
    mel[:,t_mask_start:t_mask_end] = np.random.randint(-90, -50)
        
    return mel

def flatten(mels, mfccs):

    samples = []
    for mel, mfcc in zip(mels, mfccs):

        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        sample = np.concatenate((mel_mean, mel_std, mfcc_mean, mfcc_std), axis=None)
        
        samples.append(sample)

    return samples

def reshape(mels, mfccs, full_mels):
    
    l1 = []
    l2 = []
    for mel, mfcc in zip(mels, mfccs):
            mel_mean = np.mean(mel, axis=1)
            mel_std = np.std(mel, axis=1)                    
            mfcc = mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)
            if full_mels:
                l1.append(mel)
            else:
                l1.append(np.concatenate((mel_mean, mel_std), axis=None))
            l2.append(mfcc)
    return l1, l2


def get_processed(paths, for_sklearn=True, n_augmentations=0, n_mels=64, full_mels=False, n_mfccs=20, fmax=8000, n_workers=6):

    sklearn_data = []
    keras_data = []

    target_class_list= []

    args = [(x, for_sklearn, n_augmentations, n_mels, full_mels, n_mfccs, fmax) for x in paths]

    with Pool(processes=n_workers) as pool:
        processed_samples = pool.starmap(process, args)        
        if for_sklearn:
            for s, t in processed_samples:
                sklearn_data.extend(s)
                target_class_list.extend(t)
            pool.close()         

        else:
            mfccs = []
            mels = []    
            meta = []
            for mfcc, mel, m, t in processed_samples:
                mfccs.extend(mfcc)
                mels.extend(mel)
                meta.extend(m)
                target_class_list.extend(t)
            keras_data.append(np.asarray(mfccs))
            keras_data.append(np.asarray(mels))
            keras_data.append(np.asarray(meta))
            pool.close()

    if for_sklearn:
        return np.asarray(sklearn_data), np.asarray(target_class_list)-1
    return keras_data, np.asarray(target_class_list)-1
