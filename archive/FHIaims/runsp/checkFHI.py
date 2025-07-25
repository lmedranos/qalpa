import os, sys
from os import environ, listdir, mkdir, chdir, system, remove, path
from os import getcwd
import re
import numpy as np
import shutil

wdir = sys.argv[2]
odir = sys.argv[1]
os.chdir(odir)
folders = sorted(listdir('.'))

for fdir in folders:

    if path.exists(odir+'/'+str(fdir)+'/aims.out') == True:
        try:
            lines = open(odir+'/'+str(fdir)+'/aims.out', 'r').readlines()
            ct=0
            for line in lines[-20:]:
                if " Have a nice day." in line:
                    shutil.move(odir+'/'+str(fdir)+'/',wdir)
                    print('done')
                else:
                    ct = ct + 1
            if ct == 20 or ct == 0 or ct==1:
                print(fdir)
        except Exception:
            print(fdir)

    else:
        print(fdir)
