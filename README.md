## A refactoring of [heymann's 'Neural network based spectral mask estimation for acoustic beamforming'](https://github.com/fgnt/nn-gev)

[his ppt on icassp 2016](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&ved=2ahUKEwi6h6Sw54bdAhWFZt4KHQJlCdAQFjACegQIBxAC&url=https%3A%2F%2Fsigport.org%2Fsites%2Fdefault%2Ffiles%2Ficassp_2016_1.pdf&usg=AOvVaw3DmQHWT8LFJNCLWhmyy-QB)

### references, tools

- [paper]()
- [author's git repo](https://github.com/fgnt/nn-gev)
- [A Well-Conditioned and Sparse Estimation of Covariance
and Inverse Covariance Matrices Using a Joint Penalty](https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf)
- [Xueliang's paper]()
- [xiong's paper]()
- [google's paper]()
- [higuchi's paper]()

### requirments

- python3
- pytorch
- kaldi
- matlab

### tools
- [AudioLabs's rir generator](https://github.com/ehabets/RIR-Generator)
- (https://github.com/ehabets/RIR-Generator/blob/master/rir_generator.pdf)
- [reverb challenge 2014 generator](https://reverb2014.dereverberation.com/download.html)
- CHiME3's simulation code
- gnu parallel
- sox
- https://kr.mathworks.com/matlabcentral/fileexchange/23573-csvimport


### Installation

```sh
cd your_kaldi/egs
git clone git@github.com:gogyzzz/heymann-nn-gev-bf.git
cd heymann-nn-gev-bf/s5
```

### Data preparation

```sh
mylocal/prepare_noise.sh

mylocal/prepare_wsjcam0.sh
mylocal/prepare_mixed_wsjcam0.sh

matlab -nodesktop -nosplash -r \
 "mix('ext/mixed/wsjcam0/si_dt/mixed.csv', 1024000); exit;"
 
mylocal/prepare_chime3.sh

```

### Data preparation 2



```python

# for pytorch dataset, dataloader

def wav_to_ibm(clean, noisy, channel=-1):
    return (y_psd, x_psd, n_psd, x_mask, n_mask)

# psd normalization needed? -> no. just use batchnorm





```


### Exp. results
