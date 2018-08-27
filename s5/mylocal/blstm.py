#!/usr/bin/env python

import sys
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def init_linear(m):
    if type(m) == nn.Linear:
        tc.nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class blstm_mask_estimator(nn.Module):
    
    def __init__(self, nbin=513, ncell=128, nhid=513):

        super(blstm_mask_estimator, self).__init__()

        self.nbin = nbin

        # blstm 256 0.5
        # pytorch blstm can get dropout=0.5, nlayer=1
        self.blstm1 = tc.nn.LSTM(nbin, ncell, 1,
                batch_first=True,
                bias=True,
                bidirectional=True)

        # ff 513 x 513 0.5
        self.fc2 = nn.Linear(ncell*2, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        # ff 513 x 513 0.5
        self.fc3 = nn.Linear(nhid, nhid)
        self.bn3 = nn.BatchNorm1d(nhid)

        # ff 513 x 1026
        self.s_mask_estimate = nn.Linear(nhid, nbin)
        self.n_mask_estimate = nn.Linear(nhid, nbin)

        self.apply(init_linear)

    def forward(self, y_psd):

        # bsz: 1
        y_psd = tc.reshape(y_psd, (1, -1, self.nbin))
        #print(np.shape(y_psd))
        blstm1, hn = self.blstm1(y_psd)
        #print(np.shape(blstm1))

        # squeeze needed. [1, -1, 2*ncell] -> [-1, 2*ncell]
        out2 = self.bn2(self.fc2(tc.squeeze(blstm1)))
        out3 = self.bn3(self.fc3(out2))

        return tc.sigmoid(self.s_mask_estimate(out3)),\
            tc.sigmoid(self.n_mask_estimate(out3))

if __name__ == "__main__":
    from dataloader import ibm_dataset
    from torch.utils.data import DataLoader
    import torch as tc
    import torch.nn as nn
    import numpy as np
    from tqdm import tqdm 

    
    dataset = ibm_dataset('sample/clean.scp', 'sample/noisy.scp')
    dataloader = DataLoader(dataset, 1, num_workers=1)

    model = blstm_mask_estimator()
    optim = tc.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(1):
        for y_psd, x_mask, n_mask in tqdm(dataloader, total=len(dataset)):

            # due to batchsize(1) != number of frames
            y_psd = tc.squeeze(y_psd)
            x_mask = tc.squeeze(x_mask)
            n_mask = tc.squeeze(n_mask)

            print(np.shape(y_psd))
            print(np.shape(x_mask))
            continue

            x_mask_hat, n_mask_hat = model(y_psd)

            optim.zero_grad()
            #print(np.shape(x_mask_hat))

            x_loss = F.binary_cross_entropy(x_mask_hat, x_mask)
            n_loss = F.binary_cross_entropy(n_mask_hat, n_mask)
            loss = x_loss + n_loss

            loss.backward()
            optim.step()
            # you should convert 1x1 tensor to scalar 
            #    using item(), not data[0]
            print(loss.item())
