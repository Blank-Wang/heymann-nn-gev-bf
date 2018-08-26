#!/bin/bash

# convert reverb14 noise to 1ch
noise=ext/reverb14/noise.scp
noise_orig=ext/reverb14/noise_orig.scp

paste -d ' ' $noise_orig $noise | parallel --colsep ' ' sox {} remix 1

# please prepare chime3's noise yourself. sorry

head -2 ext/*/noise.scp
# ==> ext/chime3/noise.scp <==
# ext/chime3/noise/BGD_150203_010_CAF.CH1.wav
# ext/chime3/noise/BGD_150203_010_CAF.CH2.wav
#
# ==> ext/reverb14/noise.scp <==
# ext/reverb14/noise/Noise_LargeRoom1_1.wav
# ext/reverb14/noise/Noise_LargeRoom1_10.wav

cat ext/*/noise.scp > ext/noise.scp

echo ""
echo "made ext/noise.scp"
echo ""
head -2 ext/noise.scp
wc -l ext/noise.scp
