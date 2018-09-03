






if __name__=="__main__":

    from dataloader import ibm_dataset
    from dataloader import ibm_dataset_nch
    from ff import ff_mask_estimator
    from torch.utils.data import DataLoader
    import torch as tc
    import torch.nn as nn
    import numpy as np
    import datetime as dt
    import os

    expdir = 'sample_exp/'
    score_log = expdir + '/score_log'
    nch = 6

    devset = ibm_dataset_nch('sample/clean.6ch.scp', 'sample/noisy.6ch.scp')
       
    devloader = DataLoader(devset, 1, num_workers=1)

    ff = ff_mask_estimator().load_state_dict(tc.load(expdir + '/best_state_dic.pth'))

    score_dic_list = []
    snr_csv_header = ','.join(['snr%dch'%(i) for i in range(nch)])
    
    for y_psd, x_mask, n_mask, noisypaths, cleanpaths in tqdm(devloader, total=len(devset)):

        y_psd = y_psd.reshape((-1,513))
        x_mask = x_mask.reshape((-1,513))
        n_mask = n_mask.reshape((-1,513))

        # getting speech, noise masks
        x_mask_hat, n_mask_hat = ff(y_psd)

        x_mask_hat_ch = x_mask_hat.reshape(nch, -1, 513)
        n_mask_hat_ch = n_mask_hat.reshape(nch, -1, 513)
        print(np.shape(x_mask_hat_ch))

        # getting median mask
        n_mask_hat = np.median(x_mask_hat_ch.data(), axis=0)
        x_mask_hat = np.median(n_mask_hat_ch.data(), axis=0)
        print(np.shape(x_mask_hat))

        # gev using mask (getting y_hat)
        y_hat = gevbf_with_mask(
                y_psd.reshape(nch,-1,513).data(),
                x_mask_hat,
                n_mask_hat)

        # SNR
        snrs = get_snr(y_hat,cleanpaths) 

        # other measure
        score_dic_list.append({'snr':','.join(snrs)}) 
        
        # 




    

    # SNR

    




    
