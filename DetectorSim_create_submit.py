#!/usr/bin/env python

submit = """
Universe       = vanilla
Executable     = /home/rkoirala/icerecsim/V05-00-07/build/env-shell.sh
Arguments      = /data/user/rkoirala/CORSIKA/./run_detector_simulation.sh {infile} {unzipped} {ene} {runnum} {datanum} {resampling_radius} {resampling_times} {raise_observation}

initialdir     = /data/user/rkoirala/CORSIKA/Error/
Output         = {unzipped}.{datanum}.out
Error          = {unzipped}.{datanum}.err
Log            = {unzipped}.{datanum}.log

request_memory = {request_memory}
getenv         = true
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
notification  = never

{partition}

#request_memory = 5G
#+AccountingGroup="1_week.rkoirala" 
#notify_user  = rkoirala@udel.edu

Queue
"""

import sys, os, glob
import numpy
from optparse import OptionParser

parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
parser.add_option("--dataset", "-d", type=str, help="Input Level1 i3 files")
parser.add_option("--ene", "-e", type=str, help="Run specific energy bin.", default='None')

opts, args = parser.parse_args()
datanum = args[0]

'''
Proton:
    9508  [sybill2.1, logE:4-5] 
    10410 [sybill2.1, logE:5-8]
    20252 [QGSJETII-04, logE:4-5]
    10951 [QGSJETII-04, logE:5-8]
    
Helium:
    7362  [sybill2.1, logE:4-5] 
    11663 [sybill2.1, logE:5-8]
    20253 [QGSJETII-04, logE:4-5]
    12583 [QGSJETII-04, logE:5-8]

Oxygen:
    7364  [sybill2.1, logE:4-5] 
    12605 [sybill2.1, logE:5-8]
    20254 [QGSJETII-04, logE:4-5]
    12584 [QGSJETII-04, logE:5-8]

Iron:
    9614  [sybill2.1, logE:4-5] 
    10889 [sybill2.1, logE:5-8]
    20255 [QGSJETII-04, logE:4-5]
    10954 [QGSJETII-04, logE:5-8]
'''
dataset_list= ['9508' , '7362' , '7364' , '9614' , '10410', '11663', '12605', '10889',
               '20252', '20253', '20254', '20255', '10951', '12583', '12584', '10954']
if datanum not in dataset_list:
    raise Exception("Provide proper dataset number. You entered: %s"%datanum)
    
indent_dir  = {'4.0':'1', '4.1':'2', '4.2':'3', '4.3':'4', '4.4':'5', '4.5':'6', '4.6':'7', '4.7':'8', '4.8':'9', '4.9':'0',
               '5.0':'1', '5.1':'2', '5.2':'3', '5.3':'4', '5.4':'5', '5.5':'6', '5.6':'7', '5.7':'8', '5.8':'9', '5.9':'0',
               '6.0':'1', '6.1':'2', '6.2':'3', '6.3':'4', '6.4':'5', '6.5':'6', '6.6':'7', '6.7':'8', '6.8':'9', '6.9':'0',
               '7.0':'1', '7.1':'2', '7.2':'3', '7.3':'4', '7.4':'5', '7.5':'6', '7.6':'7', '7.7':'8', '7.8':'9', '7.9':'0'}

#name: ['detectorSim', 'qgsjet']
name        = 'qgsjet'

if (datanum=='9508' or datanum=='9614'):
    ene_list = ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9']
    resampling_times = 1
    raise_observation = 3

elif (datanum=='7362' or datanum=='7364'):
    ene_list = ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9']
    resampling_times = 40
    raise_observation = 3

elif (datanum=='10410' or datanum=='10889' or datanum=='10951' or datanum=='10954'):
    ene_list = ['5.0', '5.1', '5.2', '5.3', '5.4', '5.5', '5.6', '5.7', '5.8', '5.9',
                '6.0', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8', '6.9',
                '7.0', '7.1', '7.2', '7.3']    
    resampling_times = 50
    raise_observation = 3
    run_script = 'run_detector_simulation.sh'

elif (datanum=='11663' or datanum=='12605' or datanum=='12583' or datanum=='12584'):
    ene_list = ['5.0', '5.1', '5.2', '5.3', '5.4', '5.5', '5.6', '5.7', '5.8', '5.9',
                '6.0', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8', '6.9',
                '7.0', '7.1', '7.2', '7.3']
    resampling_times = 50
    raise_observation = 0  # raise observation level by 3m not required. # alt=2837m.
    #run_script = 'run_detector_simulation_He_O.sh'

elif (datanum=='20252' or datanum=='20253' or datanum=='20254' or datanum=='20255'):
    ene_list = ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9']
    resampling_times = 1
    raise_observation = 0  # raise observation level by 3m not required. # alt=2837m.
    #run_script = 'run_detector_simulation.sh'
    
if opts.ene!='None':
    ene_list=[opts.ene]

submit_file = open('condor_'+name+'_'+datanum+'.submit', 'w')
workdir = "/data/user/rkoirala/CORSIKA/"
datadir = "/data/user/rkoirala/DATA/"+datanum+"/"

jobcount = 0
counter  = 0

filelistname = 'DetectorSim'+datanum+'_filelist.txt'
if filelistname in glob.glob('*.txt'):
    loadfi = numpy.loadtxt(filelistname, dtype='str')
else:
    loadfi = []

print loadfi

for ene in ene_list:
    # Submit 1 job for 1 binary file.
    inputfiles = glob.glob('/data/sim/IceTop/2009/generated/CORSIKA-ice-top/'+datanum+'/'+ene+'/DAT*.bz2')
    outfils = glob.glob('/data/user/rkoirala/DATA/'+datanum+'/'+ene+'/DetectorSim/DAT*.bz2')
    binfils = glob.glob('/data/user/rkoirala/DATA/'+datanum+'/'+ene+'/DAT*')
    
    if (float(ene)>=4.0) and (float(ene)<5.0):
         resampling_radius=200
         request_memory='2G'
         partition=''
    if (float(ene)>=5.0) and (float(ene)<6.0):
         resampling_radius=400
         request_memory='3G'
         partition=''
    if (float(ene)>=6.0) and (float(ene)<7.0):
         resampling_radius=600
         request_memory='4G'
         partition=''
    if (float(ene)>=7.0) and (float(ene)<8.0):
         resampling_radius=800
         request_memory='5G' #increase this to 10G if necessary.
         partition="+AccountingGroup='1_week.rkoirala'"
         
    for infile in inputfiles:
        outf    = '/data/user/rkoirala/DATA/'+datanum+'/'+ene+'/DetectorSim/'+infile[-13:-4]+'_Oct2016Snow.i3.bz2'        
        ouf     = infile[-13:-4]+'_Oct2016Snow.i3.bz2'
        binfile = '/data/user/rkoirala/DATA/'+datanum+'/'+ene+'/'+infile[-13:-4] 
        runnum  = int(infile[-8:-4])
        #if (outf not in outfils) or (binfile in binfils):
        if ouf not in loadfi:
            print ''
            print 'counter   :', counter
            print 'runnum    :', runnum
            print 'ene bin   :', ene
            print 'infile    :', infile
            print 'outfile   :', outf
            print 'datanum   :', datanum
            print ''

            filename = workdir+'CondorSubmit/submit_'+name+'_'+datanum+'_'+ene+'_%s.submit'%counter
            f = open(filename, 'w')
            f.write(submit.format(infile=infile,
                                  unzipped=infile[-13:-4],
                                  ene=ene,
                                  runnum=runnum,
                                  datanum=datanum,
                                  resampling_radius=resampling_radius,
                                  resampling_times=resampling_times,
                                  raise_observation=raise_observation,
                                  #run_script=run_script,
                                  request_memory=request_memory,
                                  partition=partition
                                  )
                    )
            f.close()
            submit_file.write('JOB job%s_%s %s\n'%(datanum, str(counter),filename))
            jobcount += 1
            counter  += 1
print 'Jobs to process: ', jobcount
submit_file.close()


