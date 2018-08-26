#!/bin/bash

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

echo ""
echo "ext/mixed/wsjcam0/*/mixed.csv created"
echo ""
head -2 ext/mixed/wsjcam0/*/mixed.csv
