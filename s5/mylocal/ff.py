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

class ff_mask_estimator(nn.Module):
    
    def __init__(self, nbin=513, nhid=513):

        super(ff_mask_estimator, self).__init__()

        self.fc1 = nn.Linear(nbin, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)

        self.fc2 = nn.Linear(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.fc3 = nn.Linear(nhid, nhid)
        self.bn3 = nn.BatchNorm1d(nhid)

        self.s_mask_estimate = nn.Linear(nhid, nbin)
        self.n_mask_estimate = nn.Linear(nhid, nbin)

        self.apply(init_linear)

    def forward(self, y_psd):

        relu1 = F.relu(self.bn1(self.fc1(y_psd)))
        ##print(np.shape(relu1))
        return tc.sigmoid(self.s_mask_estimate(relu1)), \
            tc.sigmoid(self.n_mask_estimate(relu1))

if __name__ == "__main__":
    from dataloader import ibm_dataset
    from torch.utils.data import DataLoader
    import torch as tc
    import torch.nn as nn
    import numpy as np
    from tqdm import tqdm 

    
    dataset = ibm_dataset('sample/clean.scp', 'sample/noisy.scp')
    dataloader = DataLoader(dataset, 1, num_workers=1)

    ff = ff_mask_estimator()
    optim = tc.optim.RMSprop(ff.parameters(), lr=0.001, momentum=0.9)

    for i in range(10):
        for y_psd, x_mask, n_mask in tqdm(dataloader, total=len(dataset)):

            y_psd = tc.squeeze(y_psd)
            x_mask = Variable(tc.squeeze(x_mask))
            n_mask = Variable(tc.squeeze(n_mask))

            x_mask_hat, n_mask_hat = ff(y_psd)

            optim.zero_grad()
            #print(np.shape(y_psd))
            #print(np.shape(x_mask))
            #print(np.shape(x_mask_hat))
            x_loss = F.binary_cross_entropy(x_mask_hat, x_mask)
            n_loss = F.binary_cross_entropy(n_mask_hat, n_mask)
            loss = x_loss + n_loss

            loss.backward()
            optim.step()
            print(loss.data[0])
