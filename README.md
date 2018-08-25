## A refactoring of [heymann's 'Neural network based spectral mask estimation for acoustic beamforming'](https://github.com/fgnt/nn-gev)

[his ppt on icassp 2016](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&ved=2ahUKEwi6h6Sw54bdAhWFZt4KHQJlCdAQFjACegQIBxAC&url=https%3A%2F%2Fsigport.org%2Fsites%2Fdefault%2Ffiles%2Ficassp_2016_1.pdf&usg=AOvVaw3DmQHWT8LFJNCLWhmyy-QB)

### references

- [paper]()
- [author's git repo](https://github.com/fgnt/nn-gev)
- [A Well-Conditioned and Sparse Estimation of Covariance
and Inverse Covariance Matrices Using a Joint Penalty](https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf)
- [Xueliang's paper]()
- [microsoft's paper]()
- [google's paper]()
- [higuchi's paper]()

### requirments

- python3
- pytorch
- kaldi
- matlab

### tools
- RIR simulation code (google's, CHiME3, CHiME4)

### to do
- getting WSJCAM0, CHIME3, CHiME3 noise data

- simulation (SNR, RT60, distance?)
- training, dev, eval dataset

```sh
simulate_chime3.m data/??? data/real_chime3
simulate_google.m data/??? data/simu_ggrir

ls data/simulated1 # all wavs are simulated (even if clean)
# noise.scp
# clean.scp
# noisy.scp


make_dataset.sh data/real_chime
 



fc/train.py fc/mytraincase1-1809xx/param.json
fc/print_mdl.py fc/mytraincase1-1809xx/model.pth
# [layer 1] ... 

lstm/train.py lstm/mytraincase1-1809xx/param.json
lstm/print_mdl.py lstm/mytraincase1-1809xx/model.pth
# [layer 1] ... 

ls mydir/mytraincase-1809xx/
# model.pth
# log: ((l2norm, max-l2norm) for layers, train, val, best_val loss, (max, min, mean, median, var)mse)x(for epochs)
# param.json: (model, dataset, bsz, log, lr, measure, nhid, ephs, ...)

fc/beamform.py \
  fc/mydataset/mytraincase1-1809xx/model.pth \
  data/noisy1 \
  data/fc_enhan_noisy1

lstm/beamform.py \
  lstm/mydataset/mytraincase1-1809xx/model.pth \
  data/noisy1 \
  data/lstm_enhan_noisy1

# kaldi 
# how to turn off language model's contribution?
train_dnnhmm.sh data/train exp/dnnhmm

decode.sh \
  data/noisy1 \
  exp/noisy1/decode

decode.sh \
  data/fc_enhanced_noisy1 \
  exp/fc_enhanced_noisy1/decode

decode.sh \
  data/lstm_enhanced_noisy_case1 \
  exp/lstm_enhanced_noisy_case1/decode
```

- exp result
