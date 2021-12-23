import numpy as np
import librosa as lr
import librosa.feature as lrf
from os import getpid

from multiprocessing import Pool

from sklearn.model_selection import train_test_split
import glob


def extract(path, for_sklearn, n_augmentations, n_mels, full_mels, n_mfccs, fmax):

    """
    Function extracts features, performs possible augmentation and a reshape to a wanted format specified by for_sklearn parameter

    path:               path to audio.
    for_sklearn:        Boolean.
    n_augmentations:    number of augmented samples.
    n_mels:             number of mel-bands
    full_mels:          Boolean, if True returns spectrogram if False returns row-wise mean and std vector.
    n_mfccs:            Number of mfc coefficents
    fmax:               Max frequency of mel spectrograms
    
    """

    
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
        
        sample = means_and_stds(mels, mfccs)
        sample = [np.append(s, gender) for s in sample]
        sample = [np.append(s, intensity) for s in sample]
        sample = [np.append(s, statement) for s in sample]

        return sample, target_list
    
    else:

        mels, mfccs = mean_and_reshape(mels, mfccs, full_mels)
        meta = [np.asarray([gender, intensity, statement], dtype=np.float32) for _ in range(n_augmentations+1)]       

        return mfccs, mels, meta, target_list


   
def augmentation(audio, sr, n_augmentations, n_mels, fmax):

    """
    Function applies data augmentation for audio samples.
    
    """
    
    copy_list = []
    
    for _ in range(n_augmentations):

        copy = audio.copy()
        
        # Apply gaussian noise to audio
        copy = copy + np.random.normal(0, 0.0001, size=len(audio))
        
        mel = lr.power_to_db(lrf.melspectrogram(copy, sr=sr, n_fft=512, n_mels=n_mels, fmax=fmax))
        
        mel = masker(mel, n_mels)
        
        copy_list.append(mel)
        
    return copy_list

def masker(mel, n_mels):

    """
    Applies randomized frequency and time domain masking for mel-spectrograms
    
    """

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

def means_and_stds(mels, mfccs):

    """
    Function calculates mean and standard deviation for every mel-band and mfc coefficent i.e. row-wise mean and std.

    returns array of concatenated mean and std arrays.
    
    """

    samples = []
    for mel, mfcc in zip(mels, mfccs):

        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        sample = np.concatenate((mel_mean, mel_std, mfcc_mean, mfcc_std), axis=None)
        
        samples.append(sample)

    return samples

def mean_and_reshape(mels, mfccs, full_mels):

    """
    function claculates mean and std of mel spectrograms and reshapes mfcc spectrogram to keras format.

    """
    
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

    """
    Function implements parallel feature extraction.
    
    """

    data = []
    target= []

    # Create arguments list for starmap
    args = [(x, for_sklearn, n_augmentations, n_mels, full_mels, n_mfccs, fmax) for x in paths]

    with Pool(processes=n_workers) as pool:
        res = pool.starmap(extract, args)

        if for_sklearn:
            for s, t in res:
                data.extend(s)
                target.extend(t)
            return np.asarray(data), np.asarray(target)-1        

        else:
            mfccs = []
            mels = []    
            meta = []
            for mfcc, mel, m, t in res:
                mfccs.extend(mfcc)
                mels.extend(mel)
                meta.extend(m)
                target.extend(t)
            data.append(np.asarray(mfccs))
            data.append(np.asarray(mels))
            data.append(np.asarray(meta))

            return data, np.asarray(target)-1


if __name__ == '__main__':
    
    # For debugging

    path_list = glob.glob('speech-emotion-recognition-ravdess-data/Actor_*/*')

    X_train_paths, X_test_paths = train_test_split(path_list, test_size=0.10, random_state=42)

    X_train, y_train = get_processed(X_train_paths)


        
    