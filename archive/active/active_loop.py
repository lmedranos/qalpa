import sys
import shutil
import os
from os import path
from ase.io import read, write
from os import environ, listdir, mkdir, chdir, system, remove
#import sympy
#import sympy.utilities
parent_dir = os.path.abspath('/data/horse/ws/leme420g-proj2025/qmdiff/e3mol_aqm/')
sys.path.append(parent_dir)
print(parent_dir)

os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy

code_dir=sys.argv[1]
data_dir=sys.argv[2]
work_dir=sys.argv[3]

print(data_dir)

ntar=int(sys.argv[4])
ptar=float(sys.argv[5])
print(ptar)
for iloop in range(1):

    if path.exists(work_dir+'/mod'+str(iloop)) == False:
        os.mkdir(work_dir+'/mod'+str(iloop))
    os.chdir(work_dir+'/mod'+str(iloop))

    os.system('python '+code_dir+'/data/aqm/split_data.py --file-path '+data_dir+' --out-path '+work_dir+'/mod'+str(iloop))

    os.system('python '+code_dir+'/run_train.py --conf '+work_dir+'/diffusion_aqm.yaml --save-dir '+work_dir+'/mod'+str(iloop)+' --dataset aqm_qm7x --dataset-root '+work_dir+'/mod'+str(iloop)+' --gpus 1 --batch-size 32')

#to  continue
#python $code/run_train.py --conf $work/diffusion_aqm.yaml --save-dir $work --dataset aqm_qm7x --dataset-root $data2 --gpus 2 --batch-size 32 --load-ckpt $work/run0/last.ckpt

    if path.exists(work_dir+'/mod'+str(iloop)+'/gen') == False:
        os.mkdir(work_dir+'/mod'+str(iloop)+'/gen')
    os.chdir(work_dir+'/mod'+str(iloop)+'/gen')

    os.system('python '+code_dir+'/run_evaluation_qm.py --model-path '+work_dir+'/mod'+str(iloop)+'/best_mol_stab.ckpt --save-dir '+work_dir+'/mod'+str(iloop)+'/gen --save-xyz --calculate-relax-change --calculate-props --batch-size 4 --ngraphs 10 --generate-ngraphs-valid --num-nodes '+str(ntar)+' --fixed-context '+str(ptar))
