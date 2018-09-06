#!/usr/bin/env python

import numpy as np
from numpy.linalg import solve
from scipy.linalg import eig
from scipy.linalg import eigh
import pdb


def gevbf_with_mask(y, x_mask_hat, n_mask_hat):
    """
    y: stft of noisy speech ( nch x nframe x nbin )
    x_mask_hat: ( nframe x nbin )
    n_mask_hat: ( nframe x nbin )

    return: 
    y_hat: stft of enhanced noisy speech ( nframe x nbin )

    """

    
    nch, nframe, nbin = y.shape

    x_psd = np.einsum('mtf,ltf->mlf',
            x_mask_hat * y, y.conj()) # y*x_mask: broadcasting along nch

    # normalization n_mask_hat only (why?)
    # ( nframe x nbin )
    n_mask_hat /= np.sum(n_mask_hat, axis=0, keepdims=True)
        
    # ( nch x nch x nbin )
    n_psd = np.einsum('mtf,ltf->mlf',
            n_mask_hat * y, y.conj()) # y*n_mask: broadcasting along nch

    # making noise PSD well-conditioned
    gamma = 1e-6
    scale = gamma * np.trace(n_psd) / n_psd.shape[0] # scale shape:( nbin )

    # scaled_eye = np.eye(n_psd.shape[0]) * scale # this has a dimension error
    scaled_eye = np.einsum('mn,f->mnf',np.eye(nch),scale) # expanded
    n_psd = (n_psd + scaled_eye) / (1 + gamma)

    # why does the author this? what is this?
    n_psd /= np.trace(n_psd)
    
    # beamforming vector: ( nch x nbin )
    f_gev = np.empty((n_psd.shape[0], n_psd.shape[-1]), dtype=np.complex)

    for f in range(f_gev.shape[-1]):
        try:
            eigvals, eigvecs = eigh(x_psd[...,f], n_psd[...,f])
            f_gev[...,f] = eigvecs[...,-1]

        except np.linalg.LinAlgError:
            print('LinAlg error for frequency {}'.format(f))
            # what ???
            f_gev[...,f] = np.ones((nch,)) / np.trace(n_psd[...,f]) * nch


    # phase correction
    for f in range(1, nbin):
        f_gev[...,f] *= np.exp(-1j*np.angle(
            np.sum(f_gev[...,f] * f_gev[...,f].conj(), axis=-1, keepdims=True)))

    print(f_gev.shape)

    # blind analytic normalization
    eps = 0 # ??? why 0
    nominator = np.einsum('mf,mnf,nlf,lf->f',
        f_gev.conj(), n_psd, n_psd, f_gev) 

    nominator = np.abs(np.sqrt(nominator))

    denominator = np.einsum('mf,mnf,nf->f',
        f_gev.conj(), n_psd, f_gev)

    denominator = np.abs(denominator)

    normalization = nominator / (denominator + eps)
    
    # beamforming
    y_hat = np.einsum('mf,mtf->tf', 
            f_gev.conj(), y)

    print(nominator.shape)
    print(denominator.shape)
    print(y_hat.shape)

    return y_hat

def get_snr(y_hat,cleanpaths): 

    #pdb.set_trace()
    x = np.empty((len(cleanpaths), y_hat.shape[0], y_hat.shape[1]), 
            dtype=np.complex128)

    snrs = []
    # ( nframe x nbin )
    for m, xpath in enumerate(cleanpaths):
        print(xpath[0]) # ypath is 1-dim tuple
        x[m,...] = rs.stft(
            rs.load(xpath[0], sr=16000, mono=True)[0] # due to (data, rs)
                , n_fft=1024).T # ( nbin x nframe ) -> ( nframe x nbin )

        # right?
        snrs.append(10 * np.log10(np.sum(y_hat * y_hat.conj())) \
                - 10 * np.log10(np.sum(x[m,...] * x[m,...].conj())))

    return snrs


if __name__=="__main__":

    from dataloader import ibm_dataset
    from dataloader import ibm_dataset_nch
    from torch.utils.data import DataLoader
    from ff import ff_mask_estimator
    import torch as tc
    import torch.nn as nn
    from tqdm import tqdm 
    import datetime as dt
    import os
    import librosa as rs

    expdir = 'sample_exp/'
    score_log = expdir + '/score_log'
    nch = 6

    devset = ibm_dataset_nch('sample/clean.6ch.scp', 'sample/noisy.6ch.scp')
    print(devset)
       
    devloader = DataLoader(devset, 1, num_workers=1)
    print(devloader)

    ff = ff_mask_estimator()
    ff.load_state_dict(tc.load(expdir + '/best_state_dic.pth'))
    print(ff)

    snr_csv_header = ','.join(['snr%dch'%(i) for i in range(nch)])

    with open(score_log, 'w') as f:
        print('noisypath, %s'%(snr_csv_header), file=f)
    
    for y_abs, _, _, noisypaths, cleanpaths in tqdm(devloader, total=len(devset)):

        y = np.empty(y_abs[0,...].shape, dtype=np.complex128)

        y_abs = y_abs.reshape((-1,513))

        print(y.shape)

        # ( nframe x nbin )
        for m, ypath in enumerate(noisypaths):
            print(ypath[0]) # ypath is 1-dim tuple
            #y[m,:,:] = rs.stft(
            y[m,...] = rs.stft(
                    rs.load(ypath[0], sr=16000, mono=True)[0] # due to (data, rs)
                    , n_fft=1024).T # ( nbin x nframe ) -> ( nframe x nbin )

        print(y.shape)

        # for test
        y += np.random.randn(y.shape[0],y.shape[1],y.shape[2])
        y_abs += tc.rand(y_abs.shape[0],y_abs.shape[1])

        # getting speech, noise masks
        x_mask_hat, n_mask_hat = ff(y_abs)

        x_mask_hat_ch = x_mask_hat.detach().numpy().reshape(nch, -1, 513)
        n_mask_hat_ch = n_mask_hat.detach().numpy().reshape(nch, -1, 513)
        print('x_mask_hat_ch')
        print(np.shape(x_mask_hat_ch))

        # getting median mask
        n_mask_hat = np.median(x_mask_hat_ch, axis=0)
        x_mask_hat = np.median(n_mask_hat_ch, axis=0)
        print('x_mask_hat')
        print(np.shape(x_mask_hat))

        # gev using mask (getting y_hat)
        y_hat = gevbf_with_mask(y,x_mask_hat,n_mask_hat)

        print('y_hat')
        print(np.shape(y_hat))

        # SNR
        snrs = get_snr(y_hat,cleanpaths) # expect 6 snrs for each channel
        snrs2 = get_snr(y[0,...],cleanpaths) # expect 6 snrs for each channel
        print(snrs[0])
        print(snrs2[0])
        
        with open(score_load, 'a') as f:
            print('%s'%(','.join([noisypaths[0]] + snrs)))

