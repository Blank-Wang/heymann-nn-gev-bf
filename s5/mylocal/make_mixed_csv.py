#!/usr/bin/env python
import pandas as pd
import random as rd
import sys

# e.g.
# src = ext/wsjcam0/si_dt/wav.scp
# noise = ext/noise.scp
# rec_cfg = rir/iip/8ch_linear004_room500400300_center150200100_upfs1024000
# clean = ext/mixed/wsjcam0/si_dt/clean.scp
# noisy = ext/mixed/wsjcam0/si_dt/noisy.scp
# mixed = ext/mixed/wsjcam0/si_dt/mixed.csv

if len(sys.argv) != 1+6:
    print('''
    Usage:
    mylocal/make_mixed_csv.py \\ 
        ext/wsjcam0/si_dt/wav.scp \\
        ext/noise.scp \\
        rir/iip/8ch_linear004_room500400300_center150200100_upfs1024000 \\
        ext/mixed/wsjcam0/si_dt/clean.scp \\
        ext/mixed/wsjcam0/si_dt/noisy.scp \\
        ext/mixed/wsjcam0/si_dt/mixed.csv
    ''')
    exit()

src = sys.argv[1]
noise = sys.argv[2]
rec_cfg = sys.argv[3]
clean = sys.argv[4]
noisy = sys.argv[5]
mixed = sys.argv[6]

src_df = pd.read_csv(src,sep=' ', names=['id', 'src']);
noise_df = pd.read_csv(noise,sep=' ', names=['noise']);

clean_df = pd.read_csv(clean,sep=' ', names=['id','clean'])
noisy_df = pd.read_csv(noisy,sep=' ', names=['id','noisy'])


mixed_df = pd.DataFrame(columns=['id', 'src', 'noise', 'rt60', 'src_azi', 'src_ele', 'noise_azi', 'noise_ele', 'snr', 'clean', 'noisy', 'rec_cfg'])
mixed_df.loc[:,'id'] = src_df.loc[:,'id']
mixed_df.loc[:,'src'] = src_df.loc[:,'src']
mixed_df.loc[:,'clean'] = clean_df.loc[:,'clean']
mixed_df.loc[:,'noisy'] = noisy_df.loc[:,'noisy']

snrs = {0, 5, 10, 15, 20, 25, 30}
rt60s = {0.2, 0.4, 0.6, 0.8, 1.0}
azimuths = {-60, -30, 0, 30, 60}
elevations = {30, 60, 90, 120, 150}

for i, row in enumerate(mixed_df.iterrows()):
    mixed_df.loc[i, 'noise'] = rd.choice(noise_df.loc[:,'noise'])
    mixed_df.loc[i, 'rt60'] = rd.choice(list(rt60s))

    src_azi = rd.choice(list(azimuths))
    src_ele = rd.choice(list(elevations))

    mixed_df.loc[i, 'src_azi'] = src_azi
    mixed_df.loc[i, 'src_ele'] = src_ele

    mixed_df.loc[i, 'noise_azi'] = rd.choice(list(azimuths - {src_azi}))
    mixed_df.loc[i, 'noise_ele'] = rd.choice(list(elevations - {src_ele}))

    mixed_df.loc[i, 'snr'] = rd.choice(list(snrs))

    mixed_df.loc[i, 'rec_cfg'] = rec_cfg

mixed_df.to_csv(mixed, index=False)

