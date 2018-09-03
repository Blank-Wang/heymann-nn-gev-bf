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
    from dataloader import ibm_dataset_nch
    from torch.utils.data import DataLoader
    import torch as tc
    import torch.nn as nn
    import numpy as np
    from tqdm import tqdm 
    import datetime as dt
    import os
    
    nowtime = dt.datetime.now().strftime('%m%d%H%M%S')
    expdir = 'exp_' + nowtime
    logpath = expdir + '/' + 'log'
    print('expdir:',expdir)
    os.system('mkdir %s'%(expdir))


    epoch = 0 # start epoch idx
    best_epoch = 0
    max_epochs = 5
    patience = 2

    isgpu = True

    #dataset = ibm_dataset('sample/clean.scp', 'sample/noisy.scp')
    trainset = ibm_dataset_nch('sample/clean.6ch.scp', 'sample/noisy.6ch.scp')
    devset = ibm_dataset_nch('sample/clean.6ch.scp', 'sample/noisy.6ch.scp')

    trainloader = DataLoader(trainset, 1, num_workers=1, 
            drop_last=True, pin_memory=isgpu)
    devloader = DataLoader(trainset, 1, num_workers=1,
            drop_last=True, pin_memory=isgpu)

    ff = ff_mask_estimator() if isgpu else ff_mask_estimator.to_gpu()

    optim = tc.optim.RMSprop(ff.parameters(), lr=0.001, momentum=0.9)

    mindevloss = 999999

    while (epoch < max_epochs and epoch - best_epoch < patience):

        for y_psd, x_mask, n_mask in tqdm(trainloader, total=len(trainset)):

            y_psd = y_psd.reshape((-1,513))
            x_mask = x_mask.reshape((-1,513))
            n_mask = n_mask.reshape((-1,513))

            #y_psd = tc.squeeze(y_psd)
            #x_mask = Variable(tc.squeeze(x_mask))
            #n_mask = Variable(tc.squeeze(n_mask))

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

            trainloss = loss.item()

            # validation
            
            devlosses = []

            for y_psd, x_mask, n_mask in tqdm(devloader, total=len(devset)):
                
                y_psd = y_psd.reshape((-1,513))
                x_mask = x_mask.reshape((-1,513))
                n_mask = n_mask.reshape((-1,513))
                x_mask_hat, n_mask_hat = ff(y_psd)
                x_loss = F.binary_cross_entropy(x_mask_hat, x_mask)
                n_loss = F.binary_cross_entropy(n_mask_hat, n_mask)
                loss = x_loss + n_loss
                
                devlosses.append(loss.item())
                

            avgdevloss = np.average(np.array(devlosses))
            
            if avgdevloss < mindevloss:
                mindevloss = avgdevloss
                best_epoch = epoch
                print('[%03d/%03d] best dev loss ever!\n'%(epoch, max_epochs))
                best_state_dic = ff.state_dict()

            devlossmsg = '[%03d/%03d] trainloss: %.3f'%(epoch,max_epochs,avgdevloss)
            trainlossmsg = '[%03d/%03d] devloss: %.3f'%(epoch,max_epochs,trainloss)

            print(trainlossmsg,'\n')
            print(devlossmsg,'\n')

            os.system('echo %s >> %s'%(devlossmsg, logpath))
            os.system('echo %s >> %s'%(trainlossmsg, logpath))

            epoch += 1 

    tc.save(best_state_dic, expdir + '/best_state_dic.pth')
    os.system('ls %s'%(expdir))
