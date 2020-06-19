#!/usr/bin/env python

'''
This is the first part of submitting Level1 scripts in Asterix.
This scripts creates sbatch_level1****.submit which does the following job.
     Then run run_sbatch.sh to submit the jobs in asterix.

To Run:
   [command prompt]$:  ./Level1_create_submit.py 10410
Then:
   [command prompt]$:  ./run_sbatch.sh 10410

#{partition}
#SBATCH --time=15000 runs this job for 15000 mins [>10 days].
#SBATCH --mail-user=rkoirala@udel.edu
#SBATCH --mail-type=ALL
'''

submit="""#!/bin/bash
#SBATCH --job-name={name}_{datanum}_{ene}_job{counter}
#SBATCH --output=/data/icet0/rkoirala/CORSIKA/Error/{name}_{datanum}_{ene}_{counter}.out
#SBATCH --error=/data/icet0/rkoirala/CORSIKA/Error/{name}_{datanum}_{ene}_{counter}.err
#
#SBATCH --mem-per-cpu=4096
#{partition}

sh /home/rkoirala/icerecsim/V05-00-07/build/env-shell.sh python {core_script} --gcd /data/icet0/rkoirala/GCD/GeoCalibDetectorStatus_2016.57531_V0_OctSnow.i3.gz --generator corsika-icetop --with-icetop --detector IC86 -n {resampling_times} --without-photon-prop --top-response g4 -r {counter} -o {outfile} {infile}
"""

import sys, os, glob
from optparse import OptionParser

parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
parser.add_option("--dataset", "-d", type=str, help="Input Level1 i3 files")
parser.add_option("--ene", "-e", type=str, help="Run specific energy bin.", default='None')

opts, args = parser.parse_args()
datanum = args[0]

indent_dir  = {'3.5':'6', '3.6':'7', '3.7':'8', '3.8':'9', '3.9':'0',
               '4.0':'1', '4.1':'2', '4.2':'3', '4.3':'4', '4.4':'5', '4.5':'6', '4.6':'7', '4.7':'8', '4.8':'9', '4.9':'0',
               '5.0':'1', '5.1':'2', '5.2':'3', '5.3':'4', '5.4':'5', '5.5':'6', '5.6':'7', '5.7':'8', '5.8':'9', '5.9':'0',
               '6.0':'1', '6.1':'2', '6.2':'3', '6.3':'4', '6.4':'5', '6.5':'6', '6.6':'7', '6.7':'8', '6.8':'9', '6.9':'0',
               '7.0':'1', '7.1':'2', '7.2':'3', '7.3':'4', '7.4':'5', '7.5':'6', '7.6':'7', '7.7':'8', '7.8':'9', '7.9':'0'
           }
name        = 'detectorSim'

if (datanum=='10922'):
    ene_list = ['3.5', '3.6', '3.7', '3.8', '3.9']
    resampling_times = 1

elif (datanum=='9508' or datanum=='9614'):
    ene_list = ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9']
    resampling_times = 1

elif (datanum=='7362' or datanum=='7364'):
    ene_list = ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9']
    resampling_times = 28

elif (datanum=='10410' or datanum=='11663' or datanum=='10889' or datanum=='12605'):
    #ene_list = ['5.0', '5.1', '5.2', '5.3', '5.4', '5.5', '5.6', '5.7', '5.8', '5.9',
    #            '6.0', '6.1', '6.2', '6.3', '6.4', '6.5']    
    #ene_list = ['5.1', '5.2', '5.3', '5.4', '5.5', '5.6', '5.7', '6.1', '6.2', '6.3', '6.5']
    ene_list = ['7.4']
    resampling_times = 50

if opts.ene!='None':
    ene_list=[opts.ene]

ene_list = ['7.4']
submit_file = open('sbatch_'+name+'_'+datanum+'.submit', 'w')
workdir = "/data/icet0/rkoirala/CORSIKA/"
datadir = "/data/icet0/rkoirala/DATA/Simulation/"+datanum+"/"

counter  = 1
jobcount = 0
for ene in ene_list:
    indentifier = indent_dir[ene]
    # What resampling radius to use. 4-5: 200m, 5-6:400m, 6-7:600m
    if float(ene)>=3.5 and float(ene)<5.0:
        radius = '200'
        range_num = 10
        zfill_num = 1
    elif float(ene)>=5.0 and float(ene)<6.0:
        radius = '400'
        range_num = 100
        zfill_num = 2
    elif float(ene)>=6.0 and float(ene)<6.6:
        radius = '600'
        range_num = 100
        zfill_num = 2
    elif float(ene)>=6.6 and float(ene)<7.0:
        radius = '600'
        range_num = 1000
        zfill_num = 3
    elif float(ene)>=7.0:
        radius = '800'
        range_num = 1000
        zfill_num = 3
        partition='SBATCH --partition=long'
    if (float(ene)>=6.0) and (float(ene)<=6.2): # which queue/node to use.
        partition='SBATCH --partition=C8_24'
    elif float(ene)>6.2:
        partition='SBATCH --partition=long'
    else:
        partition=''

    # Provide observation level. No need for 11663 and 12605 as observation level is 2837 m, others 2834m.
    if datanum=='11663' or datanum=='12605':
        core_script = 'fullSimulation.py --resampling_radius '+radius
    else:
        core_script = 'fullSimulation.py --resampling_radius '+radius+' --raise_observation 3'
    
    completed_detector_sim = glob.glob(datadir+ene+"/DetectorSim/DAT_*.i3.bz2")

    # One output per one input CORSIKA binary file.
    infiles = glob.glob(datadir+ene+'/DAT*')
    for infile in infiles:
        outfile=datadir+ene+"/DetectorSim/"+infile[-9:]+"_Oct2016Snow.i3.bz2"
        # If any one of these detector simulation is done previously, do not repeat it.
        #if outfile not in completed_detector_sim:
        if True:
            print ''
            print 'counter   :', counter
            print 'dataSet   :', datanum
            print 'ene bin   :', ene
            print 'resamp rad:', radius, 'm'
            print 'datadir   :', datadir
            print 'infile    :', infile
            print 'outfile   :', outfile
            print 'submit    :', workdir+'SbatchSubmit/submit_'+name+'_'+datanum+'_'+ene+'_%s.submit'%counter
            print ''
            
            filename = workdir+'SbatchSubmit/submit_'+name+'_'+datanum+'_'+ene+'_%s.submit'%counter
            f = open(filename, 'w')
            f.write(submit.format(infile=infile,
                                  outfile=outfile,
                                  datanum=datanum,
                                  ene=ene,
                                  name=name,
                                  core_script=core_script,
                                  resampling_times=resampling_times,
                                  partition=partition,
                                  counter=counter))
            f.close()
            submit_file.write('sbatch %s\n'%filename)
            jobcount += 1
            counter += 1 # counter is the run number
print 'Jobs to process: ', jobcount
submit_file.close()

