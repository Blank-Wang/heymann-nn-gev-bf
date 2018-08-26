#!/bin/bash

source path.sh

wsjcam0=mypath/LDC95S24/wsjcam0
wsj0=mypath/LDC93S6B/11-13.1

local/wsjcam0_data_prep.sh $wsjcam0 $wsj0

mkdir -p ext/wsjcam0/si_{tr,dt,et}; 

awk '{print $1,"ext/wsjcam0/si_tr/"$1".wav"}' data/local/data/si_tr_wav.scp > ext/wsjcam0/si_tr/wav.scp
awk '{print $1,"ext/wsjcam0/si_dt/"$1".wav"}' data/local/data/si_dt_wav.scp > ext/wsjcam0/si_dt/wav.scp
awk '{print $1,"ext/wsjcam0/si_et/"$1".wav"}' data/local/data/si_et_wav.scp > ext/wsjcam0/si_et/wav.scp

# maybe slow. parallelization needed
wav-copy scp:data/local/data/si_tr_wav.scp scp:ext/wsjcam0/si_tr/wav.scp 
wav-copy scp:data/local/data/si_dt_wav.scp scp:ext/wsjcam0/si_dt/wav.scp
wav-copy scp:data/local/data/si_et_wav.scp scp:ext/wsjcam0/si_et/wav.scp

imm_si_tr="ext/wsjcam0/si_tr/imm_simu"
imm_si_dt="ext/wsjcam0/si_dt/imm_simu"
imm_si_et="ext/wsjcam0/si_et/imm_simu"

mkdir -p $imm_si_tr $imm_si_dt $imm_si_et; 

awk -v dir="$imm_si_tr" '{print $1,dir"/"$1".wav"}' data/local/data/si_tr_wav.scp > $imm_si_tr/wav.scp
awk -v dir="$imm_si_dt" '{print $1,dir"/"$1".wav"}' data/local/data/si_dt_wav.scp > $imm_si_dt/wav.scp
awk -v dir="$imm_si_et" '{print $1,dir"/"$1".wav"}' data/local/data/si_et_wav.scp > $imm_si_et/wav.scp

echo ""
echo "made these files"
echo ""
head -2 $imm_si_tr/wav.scp 
head -2 $imm_si_dt/wav.scp 
head -2 $imm_si_et/wav.scp 

wc -l $imm_si_tr/wav.scp 
wc -l $imm_si_dt/wav.scp 
wc -l $imm_si_et/wav.scp 
