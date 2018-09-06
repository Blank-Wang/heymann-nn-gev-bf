#!/usr/bin/env python

# this code is modified version of 
#  https://github.com/fgnt/nn-gev/blob/master/fgnt/signal_processing.py
#  https://github.com/fgnt/nn-gev/blob/master/fgnt/mask_estimation.py


import sys
import librosa as rs
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt

def _voiced_unvoiced_split_characteristic(number_of_frequency_bins):
    split_bin = 200
    transition_width = 99
    fast_transition_width = 5
    low_bin = 4
    high_bin = 500

    a = np.arange(0, transition_width)
    a = np.pi / (transition_width - 1) * a
    transition = 0.5 * (1 + np.cos(a))

    b = np.arange(0, fast_transition_width)
    b = np.pi / (fast_transition_width - 1) * b
    fast_transition = (np.cos(b) + 1) / 2

    transition_voiced_start = int(split_bin - transition_width / 2)
    voiced = np.ones(number_of_frequency_bins)

    # High Edge
    voiced[transition_voiced_start - 1: (
        transition_voiced_start + transition_width - 1)] = transition
    voiced[transition_voiced_start - 1 + transition_width: len(voiced)] = 0

    # Low Edge
    voiced[0: low_bin] = 0
    voiced[low_bin - 1: (low_bin + fast_transition_width - 1)] = \
        1 - fast_transition

    # Low Edge
    unvoiced = np.ones(number_of_frequency_bins)
    unvoiced[transition_voiced_start - 1: (
        transition_voiced_start + transition_width - 1)] = 1 - transition
    unvoiced[0: (transition_voiced_start)] = 0

    # High Edge
    unvoiced[high_bin - 1: (len(unvoiced))] = 0
    unvoiced[
    high_bin - 1: (high_bin + fast_transition_width - 1)] = fast_transition

    return (voiced, unvoiced)

def wav_to_ibm(clean_wav_path, noisy_wav_path, 
        channel=-1,
        threshold_unvoiced_speech=5,
        threshold_voiced_speech=0,
        threshold_unvoiced_noise=-10,
        threshold_voiced_noise=-10,
        low_cut=5,
        high_cut=500):

    clean, sr = rs.load(clean_wav_path,sr=16000, mono=False)
    noisy, sr = rs.load(noisy_wav_path,sr=16000, mono=False)

    #print(np.max(clean))
    #print(np.max(noisy))

    #print(np.shape(clean))
    #print(np.shape(noisy))
    # concat channel
    if channel == -1:
        clean = np.reshape(clean,(-1,))
        noisy = np.reshape(noisy,(-1,))
    else:
        clean = clean[channel, :]
        noisy = noisy[channel, :]

    #print(np.shape(clean))
    #print(np.shape(noisy))

    # |x|^2 = Power-Spectral-Density
    x = rs.stft(clean, n_fft=1024)
    x_psd = x * x.conjugate()

    y = rs.stft(noisy, n_fft=1024)
    #y_psd = y * y.conjugate()

    n = y - x
    n_psd = n * n.conjugate()

    #print(np.shape(x_psd))
    #print(np.shape(n_psd))
    #print(np.shape(y_psd))
    
    (voiced, unvoiced) = _voiced_unvoiced_split_characteristic(x.shape[0])

    # calculate the thresholds
    threshold = threshold_voiced_speech * voiced + \
        threshold_unvoiced_speech * unvoiced
    threshold_new = threshold_unvoiced_noise * voiced + \
        threshold_voiced_noise * unvoiced

    # each frequency is multiplied with another threshold
    c = np.power(10, (threshold / 10))
    x_psd_threshold = x_psd / np.reshape(c,(-1,1))
    c_new = np.power(10, (threshold_new / 10))
    x_psd_threshold_new = x_psd / np.reshape(c_new,(-1,1))

    x_mask = (x_psd_threshold > n_psd)

    x_mask = np.logical_and(x_mask, (x_psd_threshold > 0.005))
    x_mask[0:low_cut - 1, ...] = 0
    x_mask[high_cut:len(x_mask[0]), ...] = 0

    n_mask = (x_psd_threshold_new < n_psd)

    n_mask = np.logical_or(n_mask, (x_psd_threshold_new < 0.005))
    n_mask[0: low_cut - 1, ...] = 1
    n_mask[high_cut: len(n_mask[0]), ...] = 1

    #print(np.shape(x_mask))
    #print(np.shape(n_mask))

    return (y, 
            x_psd.real, 
            n_psd.real, 
            x_mask.astype(np.float32), 
            n_mask.astype(np.float32))

if __name__ == "__main__":

    _, x_psd, n_psd, x_mask, n_mask = wav_to_ibm(sys.argv[1], sys.argv[2],
            channel=0)

    data = [x_psd, n_psd, x_mask, n_mask]
    title = ['x_psd', 'n_psd', 'x_mask', 'n_mask']

    fig = plt.figure(figsize=[8, 8])
    for i in range(4):
        #print(np.shape(data[i]))
        plt.subplot(2, 2, i+1)
        data[i] = data[i].astype(float)
        display.specshow(rs.amplitude_to_db(
            data[i],ref=np.max),
            y_axis='linear', x_axis='time',
            sr=16000)
        plt.title(title[i])

    png = 'psd_ibm_%s.png'%(sys.argv[1].split('/')[-1].split('.')[0])
    #print(png)
    fig.savefig(png)
