#!/usr/bin/env python

import sys, os, glob, numpy
import time
import commands

datadir  = "/data/icet0/rkoirala/DATA/Experiment/2016/2016ML/"
allfiles = glob.glob(datadir+'Run*_ML_slcQcut.h5')

print 'Total jobs to run:', len(allfiles)

submit="""#!/bin/bash
#
#SBATCH --job-name={run}
#SBATCH --output=/data/icet0/rkoirala/CORSIKA/Error/{run}.out
#SBATCH --error=/data/icet0/rkoirala/CORSIKA/Error/{run}.err
#
python spark_predict_data.py -f {infile} --which_half 1st
"""
# python spark_predict_data.py -f {infile} --which_half 2nd
# python spark_predict_data.py -f {infile} --which_half 1st --isqgsjet
# python spark_predict_data.py -f {infile} --which_half 2nd --isqgsjet

sbatchfn = 'sbatch_prepare_data.submit'
submit_file = open(sbatchfn, 'w')
for i, hfname in enumerate(allfiles):
    run      = hfname[-23:-14]
    filename = '/data/icet0/rkoirala/CORSIKA/SbatchSubmit/submit_%s.submit'%run
    
    f = open(filename, 'w')
    f.write(submit.format(infile=hfname, run=run))
    
    submit_file.write('sbatch %s\n'%filename)
    counter += 1
    f.close()
        
submit_file.close()


# Now submit the job that has been created.
with open(sbatchfn, "r") as ins:
    for line in ins:
        os.system(line) # run 1 job per run.
        time.sleep(1)
