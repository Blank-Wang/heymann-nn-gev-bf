#!/usr/bin/env python 

import sys
import numpy as np
import torch as tc
from torch.utils.data import Dataset
import csv

from wav_to_ibm import wav_to_ibm 
#if __name__ != "__main__":
#    from mylocal.wav_to_ibm import wav_to_ibm

class ibm_dataset(Dataset):

    def __init__(self, clean_wavs_path, noisy_wavs_path):

        # expected <utter_id> <wavpath>
        with open(clean_wavs_path) as f:
            reader = csv.reader(f, delimiter=' ')
            self.clean_id_wavs = [l for l in reader]

        with open(noisy_wavs_path) as f:
            reader = csv.reader(f, delimiter=' ')
            self.noisy_id_wavs = [l for l in reader]

        # check id
        for cid, nid in zip(self.clean_id_wavs, self.noisy_id_wavs):
            if cid[0] != nid[0]:
                #print('in %s and %s'%(clean_wavs_path, noisy_wavs_path))
                #print('%s != %s'%(cid,nid))
                return

    def __len__(self):
        return len(self.noisy_id_wavs)

    def __getitem__(self, idx):

        y, _, _, x_mask, n_mask = wav_to_ibm(
                self.clean_id_wavs[idx][1], self.noisy_id_wavs[idx][1])
        #print(np.shape(y))
        #print(np.shape(x_mask))

        # (nbin x nframe) -> (nframe x nbin): nframe is batchsize
        ##print(np.shape(tc.FloatTensor(y.T)))
        return \
                tc.FloatTensor(y.T), \
                tc.FloatTensor(x_mask.T), \
                tc.FloatTensor(n_mask.T), \
                [self.noisy_id_wavs[new_idx-i][1] for i in range(self.nch)], \
                [self.clean_id_wavs[new_idx-i][1] for i in range(self.nch)]

# it expects (default: 6ch) 
# A_CH1 /mypath/A.CH1.wav
# A_CH2 /mypath/A.CH2.wav
# A_CH3 /mypath/A.CH3.wav
# A_CH4 /mypath/A.CH4.wav
# A_CH5 /mypath/A.CH5.wav
# A_CH6 /mypath/A.CH6.wav
# B_CH1 /mypath/B.CH1.wav
# ...
class ibm_dataset_nch(Dataset):

    def __init__(self, clean_wavs_path, noisy_wavs_path, nch=6):

        self.nch = nch

        # expected <utter_id> <wavpath>
        with open(clean_wavs_path) as f:
            reader = csv.reader(f, delimiter=' ')
            self.clean_id_wavs = [l for l in reader]

        with open(noisy_wavs_path) as f:
            reader = csv.reader(f, delimiter=' ')
            self.noisy_id_wavs = [l for l in reader]

        # check id
        for cid, nid in zip(self.clean_id_wavs, self.noisy_id_wavs):
            if cid[0] != nid[0]:
                #print('in %s and %s'%(clean_wavs_path, noisy_wavs_path))
                #print('%s != %s'%(cid,nid))
                return
        

    def __len__(self):
        return np.uint32(len(self.noisy_id_wavs)/self.nch)

    def __getitem__(self, idx):

        y_abs_list = []
        x_mask_list = []
        n_mask_list = []

        for i in range(self.nch):
            new_idx = np.uint32(idx * self.nch + i)
            y, _, _, x_mask, n_mask = wav_to_ibm(
                self.clean_id_wavs[new_idx][1], self.noisy_id_wavs[new_idx][1],
                channel=1)
            y_abs_list.append(np.abs(y.T)) # y.T: [nframe x nbin]
            x_mask_list.append(x_mask.T)
            n_mask_list.append(n_mask.T)

        print(np.shape(np.array(y_abs_list))) # expect [nch x nframe x nbin]
        print(np.shape(y))
        #print(np.shape(x_mask))


        return \
                tc.FloatTensor(np.array(y_abs_list)), \
                tc.FloatTensor(np.array(x_mask_list)), \
                tc.FloatTensor(np.array(n_mask_list)), \
                [self.noisy_id_wavs[new_idx-i][1] for i in range(self.nch)], \
                [self.clean_id_wavs[new_idx-i][1] for i in range(self.nch)]

if __name__ == "__main__":
    # expected
    #ibm_ds = ibm_dataset('sample/clean.scp','sample/noisy.scp')
    ibm_ds = ibm_dataset_nch('sample/clean.6ch.scp','sample/noisy.6ch.scp')
    #ibm_ds = ibm_dataset(sys.argv[1], sys.argv[2])
    #ibm_ds = ibm_dataset_nch('ext/chime3/div4/dt05_clean.scp',
    #        'ext/chime3/div4/dt05_noisy.scp')
    
    tmp = ibm_ds[0]
    print(np.shape(tmp[0]))
    print(np.shape(tmp[1]))
    print(np.shape(tmp[2]))
    print(tmp[3])
    print(tmp[4])


    tmp = ibm_ds[1]
    print(np.shape(tmp[0]))
    print(np.shape(tmp[1]))
    print(np.shape(tmp[2]))
    print(tmp[3])
    print(tmp[4])
