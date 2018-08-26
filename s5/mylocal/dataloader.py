#!/usr/bin/env python 

import sys
import numpy as np
import torch as tc
from torch.autograd import Variable
from torch.utils.data import Dataset
import csv

if __name__ != "__main__":
    from mylocal.wav_to_ibm import wav_to_ibm

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
                print('in %s and %s'%(clean_wavs_path, noisy_wavs_path))
                print('%s != %s'%(cid,nid))
                return

    def __len__(self):
        return len(self.noisy_id_wavs)

    def __getitem__(self, idx):

        y_psd, _, _, x_mask, n_mask = wav_to_ibm(
                self.clean_id_wavs[idx][1], self.noisy_id_wavs[idx][1])

        # (nbin x nframes) -> (nframes x nbin): nframes is batchsize
        return \
                tc.FloatTensor(y_psd.T), \
                tc.FloatTensor(x_mask.T), \
                tc.FloatTensor(n_mask.T)



if __name__ == "__main__":
    from wav_to_ibm import wav_to_ibm 
    # expected
    # ibm_ds = ibm_dataset('sample/clean.scp','sample/noisy.scp')
    ibm_ds = ibm_dataset(sys.argv[1], sys.argv[2])
    print(ibm_ds[0])

