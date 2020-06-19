#!/usr/bin/env python

import time

submit="""#!/bin/bash
#
#SBATCH --job-name={run}
#SBATCH --output=/data/icet0/rkoirala/CORSIKA/Error/{run}.out
#SBATCH --error=/data/icet0/rkoirala/CORSIKA/Error/{run}.err
#
python sklearn_predict_energy_h4aOrig.py --isExp --savehdf --usesavedmodel --do predict -f {hfname}
"""

import sys, os, glob, numpy

datadir  = "/data/icet0/rkoirala/DATA/Experiment/2016/2016ML/"
allfiles = glob.glob(datadir+'Run*_ML_slcQcut.h5')

print 'Total jobs to run:', len(allfiles)
'''
for i, hfname in enumerate(allfiles):
    print hfname, '....', i
    #os.system("python spark_predict_data_mc.py --isExp -f "+hfname)
    #os.system("python sklearn_predict_energy.py --isExp --savehdf --usesavedmodel --do predict -f "+hfname)
    #os.system("python sklearn_predict_energy_vemmin.py --isExp --savehdf --usesavedmodel --do predict -f "+hfname)
    os.system("python sklearn_qgsjet_gridsearch.py --isExp --savehdf --usesavedmodel --do predict -f "+hfname)
    print ''
'''        
#submit_file = open('predict_energy_data.submit', 'w')
for i, hfname in enumerate(allfiles):
    filename  = 'SbatchSubmit/submit_Run%i.submit'%i
    print i, hfname
    f = open(filename, 'w')
    f.write(submit.format(hfname=hfname, run=str(i)))
    
    f.close()
    #submit_file.write('JOB jobRun%s %s\n'%(str(i), filename))
    os.system("sbatch %s"%filename)
    time.sleep(1)

#submit_file.close()
