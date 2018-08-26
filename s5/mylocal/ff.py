#!/usr/bin/env python

import sys
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def init_linear(m):
    if type(m) == nn.Linear:
        tc.nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)

class dnn(nn.Module):
    
    def __init__(self, nbin=513, nhid=513):

        super(dnn, self).__init__()

        self.fc1 = nn.Linear(nbin, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)

        self.fc2 = nn.Linear(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.fc3 = nn.Linear(nhid, nhid)
        self.bn3 = nn.BatchNorm1d(nhid)

        self.s_mask_estimate = nn.Linear(nhid, nbin)
        self.n_mask_estimate = nn.Linear(nhid, nbin)

        self.apply(init_linear)

    def forward(self, y_psd, x_mask, n_mask):

        relu1 = F.relu(self.fc1(y_psd))
        return F.sigmoid(self.s_mask_estimate(relu1)), \
            F.sigmoid(self.n_mask_estimate(relu1))

if __name__ == "__main__":
    from wav_to_ibm import wav_to_ibm
    from dataloader import ibm_dataset


