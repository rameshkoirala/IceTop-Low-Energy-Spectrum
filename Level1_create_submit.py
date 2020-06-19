#!/usr/bin/env python

'''
This is the first part of submitting Level1 scripts in Asterix.
This scripts creates sbatch_level1****.submit which does the following job.
     Then run run_sbatch.sh to submit the jobs in asterix.

To Run:
   [command prompt]$:  ./Level1_create_submit.py 10410
Then:
   [command prompt]$:  ./run_sbatch_level1.sh 10410

#SBATCH --mem-per-cpu=3072
'''

submit="""#!/bin/bash
#
#SBATCH --job-name=level1_{datanum}_job{counter}
#SBATCH --output=/data/icet0/rkoirala/CORSIKA/Error/level1_{datanum}_{ene}_{counter}.out
#SBATCH --error=/data/icet0/rkoirala/CORSIKA/Error/level1_{datanum}_{ene}_{counter}.err
#

sh /home/rkoirala/icerecdist/V05-00-05/build/env-shell.sh python SimulationFiltering.py -i {infile} -o {outfile} -g {gcdfile} --sdst-archive
"""

import sys, os, glob
from optparse import OptionParser

parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
opts, args = parser.parse_args()

datanum = args[0]
print "datanum", datanum

submit_file = open('sbatch_level1_'+datanum+'.submit', 'w')

workdir = "/data/icet0/rkoirala/CORSIKA/"
datadir = "/data/icet0/rkoirala/DATA/Simulation/"+datanum+"/"
gcdfile = "/data/icet0/rkoirala/GCD/GeoCalibDetectorStatus_2016.57531_V0_OctSnow.i3.gz"

if (datanum=='10922'):
    ene_list = ['3.5', '3.6', '3.7', '3.8', '3.9']
elif (datanum=='9508' or datanum=='7362' or datanum=='7364' or datanum=='9614' or datanum=='20252' or datanum=='20253' or datanum=='20254' or datanum=='20255'):
    ene_list = ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9']
elif (datanum=='10410' or datanum=='11663' or datanum=='12605' or datanum=='10889' or datanum=='10951' or datanum=='12583' or datanum=='12584' or datanum=='10954'):
    ene_list = ['5.0', '5.1', '5.2', '5.3', '5.4', '5.5', '5.6', '5.7', '5.8', '5.9',
                '6.0', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8', '6.9',
                '7.0', '7.1', '7.2', '7.3']
else:
    raise Exception('Please specify the proper Dataset number. Ex: ./Level1_create_submit.py 11663')

#ene_list = ['7.0', '7.1', '7.2', '7.3']

indent_dir  = {'3.5':'6', '3.6':'7', '3.7':'8', '3.8':'9', '3.9':'0',
               '4.0':'1', '4.1':'2', '4.2':'3', '4.3':'4', '4.4':'5', '4.5':'6', '4.6':'7', '4.7':'8', '4.8':'9', '4.9':'0',
               '5.0':'1', '5.1':'2', '5.2':'3', '5.3':'4', '5.4':'5', '5.5':'6', '5.6':'7', '5.7':'8', '5.8':'9', '5.9':'0',
               '6.0':'1', '6.1':'2', '6.2':'3', '6.3':'4', '6.4':'5', '6.5':'6', '6.6':'7', '6.7':'8', '6.8':'9', '6.9':'0',
               '7.0':'1', '7.1':'2', '7.2':'3', '7.3':'4', '7.4':'5'}

counter = 1
for ene in ene_list:
    if (float(ene)>=4.0) and (float(ene)<6.5):
         num_output_level1 = 10
    if (float(ene)>=6.5) and (float(ene)<8.0):
         num_output_level1 = 10
         
    inputfiles = glob.glob(datadir+ene+'/DetectorSim/DAT*.i3.bz2')
    print 'Total number of files to process:', len(inputfiles), ene
    indentifier = indent_dir[ene]
    # Divide entire DetectorSim runs into 10 Level1 files for each energy bin.
    counter = 0
    for ind in range(num_output_level1):
        ind_str = str(ind)
        ind_str = ind_str.zfill(len(str(num_output_level1 - 1)))
        infile = datadir+ene+'/DetectorSim/DAT*'+ind_str+indentifier+'_Oct2016Snow.i3.bz2'
        len_infile = len(glob.glob(infile))
        if len_infile>0:
            print counter, len_infile, infile
            filename = workdir+'SbatchSubmit/submit_level1_'+datanum+"_"+ene+'_%s.submit'%counter
            f = open(filename, 'w')
            f.write(submit.format(infile="'"+infile+"'",
                                  outfile=datadir+ene+"/Level1/Level1_"+datanum+"_"+ind_str+"_Oct2016Snow"+'.i3.bz2',
                                  gcdfile=gcdfile,
                                  datanum=datanum,
                                  counter=counter,
                                  ene=ene))
            f.close()
            submit_file.write('sbatch %s\n'%filename)
            counter += 1
        
submit_file.close()
