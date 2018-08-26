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
source path.sh

# preparation xiong paper's dataset (but I used noises from chime3 and reverb14 both)
# training: simulated wsjcam0(kaldi reverb recipe) si_tr (8ch, 7861) (reverb_noise 0-30dB) (RT60 0.1-1.0s) 
# dev: si_dt? maybe

wsjcam0=mypath/LDC95S24/wsjcam0
wsj0=mypath/LDC93S6B/11-13.1

local/wsjcam0_data_prep.sh $wsjcam0 $wsj0

# simulation with AudioLabs's image method rir

mkdir -p ext/wsjcam0/si_{tr,dt,et}; 
awk '{print $1,"ext/wsjcam0/si_tr/"$1".wav"}' data/local/data/si_tr_wav.scp > ext/wsjcam0/si_tr/wav.scp
awk '{print $1,"ext/wsjcam0/si_dt/"$1".wav"}' data/local/data/si_dt_wav.scp > ext/wsjcam0/si_dt/wav.scp
awk '{print $1,"ext/wsjcam0/si_et/"$1".wav"}' data/local/data/si_et_wav.scp > ext/wsjcam0/si_et/wav.scp

wav-copy scp:data/local/data/si_tr_wav.scp scp:ext/wsjcam0/si_tr/wav.scp # maybe slow. parallelization needed
wav-copy scp:data/local/data/si_dt_wav.scp scp:ext/wsjcam0/si_dt/wav.scp # maybe slow.
wav-copy scp:data/local/data/si_et_wav.scp scp:ext/wsjcam0/si_et/wav.scp # maybe slow.

imm_si_tr="ext/wsjcam0/si_tr/imm_simu"
imm_si_dt="ext/wsjcam0/si_dt/imm_simu"
imm_si_et="ext/wsjcam0/si_et/imm_simu"

mkdir -p $imm_si_tr $imm_si_dt $imm_si_et; 
awk -v dir="$imm_si_tr" '{print $1,dir"/"$1".wav"}' data/local/data/si_tr_wav.scp > $imm_si_tr/wav.scp
awk -v dir="$imm_si_dt" '{print $1,dir"/"$1".wav"}' data/local/data/si_dt_wav.scp > $imm_si_dt/wav.scp
awk -v dir="$imm_si_et" '{print $1,dir"/"$1".wav"}' data/local/data/si_et_wav.scp > $imm_si_et/wav.scp

# prepare noise

# convert reverb14 noise to 1ch
noise=ext/reverb14/noise.scp
noise_orig=ext/reverb14/noise_orig.scp
paste -d ' ' $noise_orig $noise | parallel --colsep ' ' sox {} remix 1

head -2 ext/*/noise.scp
# ==> ext/chime3/noise.scp <==
# ext/chime3/noise/BGD_150203_010_CAF.CH1.wav
# ext/chime3/noise/BGD_150203_010_CAF.CH2.wav
#
# ==> ext/reverb14/noise.scp <==
# ext/reverb14/noise/Noise_LargeRoom1_1.wav
# ext/reverb14/noise/Noise_LargeRoom1_10.wav

cat ext/*/noise.scp > ext/noise.scp

awk '{print $1,"ext/mixed/wsjcam0/si_dt/noisy/"$1".wav"}' ext/wsjcam0/si_dt/wav.scp > ext/mixed/wsjcam0/si_dt/noisy.scp
awk '{print $1,"ext/mixed/wsjcam0/si_et/noisy/"$1".wav"}' ext/wsjcam0/si_et/wav.scp > ext/mixed/wsjcam0/si_et/noisy.scp
awk '{print $1,"ext/mixed/wsjcam0/si_tr/noisy/"$1".wav"}' ext/wsjcam0/si_tr/wav.scp > ext/mixed/wsjcam0/si_tr/noisy.scp
awk '{print $1,"ext/mixed/wsjcam0/si_dt/clean/"$1".wav"}' ext/wsjcam0/si_dt/wav.scp > ext/mixed/wsjcam0/si_dt/clean.scp
awk '{print $1,"ext/mixed/wsjcam0/si_et/clean/"$1".wav"}' ext/wsjcam0/si_et/wav.scp > ext/mixed/wsjcam0/si_et/clean.scp
awk '{print $1,"ext/mixed/wsjcam0/si_tr/clean/"$1".wav"}' ext/wsjcam0/si_tr/wav.scp > ext/mixed/wsjcam0/si_tr/clean.scp
 
for data in si_dt si_et si_tr; do
  mylocal/make_mixed_csv.py \
    ext/wsjcam0/$data/wav.scp \
    ext/noise.scp \
    rir/iip/8ch_linear004_room500400300_center150200100_upfs1024000 \
    ext/mixed/wsjcam0/$data/clean.scp \
    ext/mixed/wsjcam0/$data/noisy.scp \
    ext/mixed/wsjcam0/$data/mixed.csv # out file
done

# I made 125(x 8) cases of rir (each case corresponding to 8ch mic array)
# rt60:(0.2, 0.4, 0.6, 0.8, 1.0) x azimuth(-60, -30, 0, 30, 60) x elevation(150, 120, 90, 60, 30)

# mylocal/make_rir.m # generate_mic_array_rir'm is my lab's script. sorry...
for data in si_dt si_et si_tr; do
  mix.m ext/mixed/wsjcam0/$data/mixed.csv
done

? origScale of before filtered <-> maxScale ?
? why noise isn't filtering ?
? scaling with magnitude ?
? why energyTarget = sqrt(sum(s{1,1}.^2)); energyInterference = sqrt(sum(interference.^2)); is energy?
? why .. wsjcam0 is noisy


# to do

make filter cell -> mat
modify filter{ } -> filter( )



in chime3... 
 ysimu=sqrt(SNR/sum(sum(ysimu.^2))*sum(sum(n.^2)))*ysimu;
xsimu=ysimu+n;



simulate_noisy.m rir/imm_clean.mat data/local/data/si_tr_wav.scp data/wsjcam0_si_tr
simulate_noisy.m rir/imm_clean.mat data/local/data/si_dt_wav.scp data/wsjcam0_si_dt





# chime3 simulation



simulate_chime3.m mypath/CHiME3/data mystorage/CHiME3/data

# heymann (same as xueliang maybe)
# training: chime3 simulated training set (Clean, Noise) <- from modified CHiME3_simulate_data.m
# dev: chime3 simulated dev set (Clean, Noise) <- from modified CHiME3_simulate_data.m

# xueliang
# training: chime3 simulated training set (6ch, 7138)
# dev: chime3 simulated dev set (6ch, 4env, 1640(410 x 4))

make_chime3_data.sh mystorage/CHiME3/data data/chime3

ls data/chime3
# dt05_simu dt05_real et05_simu et05_real tr05_simu

ls data/chime3/tr05_simu
# noise.wav.scp
# clean.wav.scp
# noisy.wav.scp

make_dataset.sh data/chime3/tr05_simu
make_dataset.sh data/chime3/dt05_simu

ls data/chime3/tr05_simu
# noise.wav.scp
# clean.wav.scp
# noisy.wav.scp
# noisy_abs_spectrum.ark noisy_abs_spectrum.scp
# ibm_x.ark ibm_x.scp
# ibm_n.ark ibm_n.scp

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
