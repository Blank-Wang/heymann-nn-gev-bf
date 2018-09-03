#!/bin/bash

mychime3=/home/haeyong/chime3/nnbf/

for mixcase in max_norm div4; do
for data in tr05 dt05 et05; do 

	mkdir -p ext/chime3/$mixcase/

	find $mychime3/isolated_$mixcase/$data*/ -type f -name '*CH?.wav' \
		| sort > path.tmp;

	awk -F/ '{print $NF,$0}' path.tmp | awk -F. '{print $1"_"$2}' > uttid.tmp
	paste -d ' ' uttid.tmp path.tmp > ext/chime3/$mixcase/"$data"_noisy.scp

	head -1 path.tmp
	head -1 uttid.tmp
	head -1 ext/chime3/$mixcase/"$data"_noisy.scp

	find $mychime3/isolated_$mixcase/$data*/ -type f -name '*Noise.wav' \
		| sort > path.tmp;

	awk -F/ '{print $NF,$0}' path.tmp | awk -F. '{print $1"_"$2}' > uttid.tmp
	paste -d ' ' uttid.tmp path.tmp > ext/chime3/$mixcase/"$data"_noise.scp

	head -1 path.tmp
	head -1 uttid.tmp
	head -1 ext/chime3/$mixcase/"$data"_noise.scp

	find $mychime3/isolated_$mixcase/$data*/ -type f -name '*Clean.wav' \
		| sort > path.tmp;

	awk -F/ '{print $NF,$0}' path.tmp | awk -F. '{print $1"_"$2}' > uttid.tmp
	paste -d ' ' uttid.tmp path.tmp > ext/chime3/$mixcase/"$data"_clean.scp

	head -1 path.tmp
	head -1 uttid.tmp
	head -1 ext/chime3/$mixcase/"$data"_clean.scp
done
done
rm path.tmp uttid.tmp
