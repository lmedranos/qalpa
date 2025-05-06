#!/bin/bash

#work=/project/scratch/p200243/Medrano/julian/qm7x-aqm
work=/project/scratch/p200243/Medrano/julian/qm7x-aqm-neq6
#ps=egap-pol
#ps=egap-dip
#ps=mbd-pol
ps=egap-pol-mbd

mkdir $work/$ps/json

ev=eval3

mkdir $work/$ps/json/$ev

for ii in {1..5}; do

	echo "$ps/$ev/tar$ii"
	
	dir1=$work/$ps/json/$ev/tar$ii/
	mkdir $dir1

	python3 gen-json.py $work/$ps/done/$ev/tar$ii/ $dir1

done
