import sys
import shutil
import os
from os import path
from os import environ, listdir, mkdir, chdir, system, remove
import numpy as np

wdir = '/project/scratch/p200243/Medrano/anatole/code/jsonfiles/'

xyz_pre = sys.argv[1]
ofile = sys.argv[2]
os.chdir(xyz_pre)
folders = sorted(listdir('.'))

for fdir in folders:
    if path.exists(ofile+fdir) == False:
        mkdir(ofile+fdir)
    chdir(ofile+fdir)

    print(fdir)
    cjson = 'python3 '+wdir+'parserFHItoJSON.py '+xyz_pre+fdir+'/aims.out'
    system(cjson)

    shutil.copy2('file.json', ofile+fdir+'/'+str(fdir)+'.json')
    remove('file.json')


