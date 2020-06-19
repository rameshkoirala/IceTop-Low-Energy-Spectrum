#!/usr/bin/env python

'''
This is the first part of submitting Level2 scripts in Asterix.
This scripts creates sbatch_level1****.submit which does the following job.
     Then run run_sbatch.sh to submit the jobs in asterix.
To Run:
   [command prompt]$:  ./Level2_create_submit.py 10410
Then:
   [command prompt]$:  ./run_sbatch_level2.sh 10410 

/home/rkoirala/icerecdist/V05-02-00/build/env-shell.sh

Note:
    If interaction model is QGSJETII-04, use the option --isqgsjet
'''
submit="""#!/bin/bash
#
#SBATCH --job-name=L2_{datanum}_{counter}
#SBATCH --output=/data/icet0/rkoirala/CORSIKA/Error/level2_{datanum}_{counter}.out
#SBATCH --error=/data/icet0/rkoirala/CORSIKA/Error/level2_{datanum}_{counter}.err
#
sh {environment} python {script} -i {infile} -g {gcdfile} -H {hdfoutput} --particle {particle} -o {outfile} {useqgsjet}
"""

import sys, os, glob, numpy
import sys, os, glob
from optparse import OptionParser

# Supply dataSet number as an argument. Ex: 10410, 11663 etc
parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
parser.add_option('--smallshower', action="store_true", help="Do small shower recontruction.")
parser.add_option('--isqgsjet', action="store_true", help="QGSJETII-04 interaction model if called")
parser.add_option('--isML', action="store_true", help="Create hdf file for ML approach, do not do Laputop reconstruction.")

opts, args = parser.parse_args()

name   = "_HLCCoreSeed_slcQcut"

if opts.smallshower:
    print "Using SmallShower Laputop Script"
    environment='/home/rkoirala/icerecdist/V05-01-03/build/env-shell.sh'
    script='Level2_reconstruction_smallshower.py'
    name+='_SmallShower'
else:
    print "For smallshower do: ./Level2_create_submit.py 10410 --smallshower"
    if opts.isML:
        environment='/home/rkoirala/icerecdist/V05-02-00/build/env-shell.sh'
        script='Level2_ML_MC.py'
        name  += '_ML'
        emin  = 5.9 # How many output Level2 hdf file? 1 output hdf file is produced below emin.
        sbatch_memory = 'SBATCH --mem-per-cpu=5G'
    else:
        environment='/home/rkoirala/icerecdist/TankWise/build/env-shell.sh'
        script= 'Level2_reconstruction_tankwise_SRT.py'
        name  += ''
        emin  = 5.9 # Below emin, 1 output hdf file, above emin 10 or 100 output Level2 hdf file.

if opts.isqgsjet:
    useqgsjet = "--isqgsjet"
    name     += "_isqgsjet"
else:
    useqgsjet = ""

datanum = args[0]
counter = 0
workdir = "/data/icet0/rkoirala/CORSIKA/"
datadir = "/data/icet0/rkoirala/DATA/Simulation/"+datanum+"/"
gcdfile = "/data/icet0/rkoirala/GCD/GeoCalibDetectorStatus_2016.57531_V0_OctSnow.i3.gz"

print "datanum", datanum

if (datanum=='10922'):
    ene_list = ['3.5', '3.6', '3.7', '3.8', '3.9']
elif (datanum=='9508' or datanum=='7362' or datanum=='7364' or datanum=='9614' or datanum=='20252' or datanum=='20253' or datanum=='20254' or datanum=='20255'):
    ene_list = ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9']
elif (datanum=='10410' or datanum=='11663' or datanum=='12605' or datanum=='10889' or datanum=='10951' or datanum=='12583' or datanum=='12584' or datanum=='10954'):
    ene_list = ['5.0', '5.1', '5.2', '5.3', '5.4', '5.5', '5.6', '5.7', '5.8', '5.9', 
                '6.0', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8', '6.9',
                '7.0', '7.1', '7.2', '7.3']
else:
    raise Exception('Please specify the proper Dataset number. Ex: ./Level2_create_submit.py 11663')
                
submit_file = open('sbatch_level2_'+datanum+'.submit', 'w')

particle_list = {'9508' :'proton',
                 '10410':'proton',
                 '7362' :'helium',
                 '11663':'helium',
                 '7364' :'oxygen',
                 '12605':'oxygen',
                 '9614' :'iron',
                 '10889':'iron',
                 '20252' :'proton',
                 '10951':'proton',
                 '20253' :'helium',
                 '12583':'helium',
                 '20254' :'oxygen',
                 '12584':'oxygen',
                 '20255' :'iron',
                 '10954':'iron'
                 }

particle = particle_list[datanum]
                
for ene in ene_list:
    # 1 Level2 output hdf file for each Level1 energy bins.
    if float(ene)<emin:
        infile = "'"+datadir+ene+'/Level1/*.i3*'+"'"
        hdfoutput = datadir+ene+"/Level2/Level2_"+datanum+"_"+ene+"_Background_SRT"+name+".h5"
        outfile = datadir+ene+"/Level2/Level2_"+datanum+"_"+ene+"_SRT.i3.bz2"

        print ene, infile, hdfoutput
        filename = workdir+'SbatchSubmit/submit_level2_'+datanum+'_%s.submit'%counter
        f = open(filename, 'w')
        f.write(submit.format(infile=infile,
                              gcdfile=gcdfile,
                              hdfoutput=hdfoutput,
                              outfile=outfile,
                              datanum=datanum,
                              particle=particle,
                              counter=counter,
                              environment=environment,
                              useqgsjet=useqgsjet,
                              script=script))
        submit_file.write('sbatch %s\n'%filename)
        counter += 1
        f.close()
        
    if float(ene)>=emin:
        num_output = len(glob.glob(datadir+ene+'/Level1/*.i3*'))
        if num_output>10:
            num_zfill = 2
        if num_output<=10:
            num_zfill = 1
        # 1 output hdf for 1 input Level1 files. 6-7 10 Level1 files. >=7 100 Level1 files.
        for i in range(num_output):
            ind = str(i)
            ind = ind.zfill(num_zfill)
            infile = "'"+datadir+ene+'/Level1/*_'+ind+'_*.i3*'+"'"
            hdfoutput = datadir+ene+"/Level2/Level2_"+datanum+"_"+ene+"_Background_SRT"+name+"_"+ind+".h5"
            outfile = datadir+ene+"/Level2/Level2_"+datanum+"_"+ene+"_SRT"+ind+".i3.bz2"

            print ene, infile, hdfoutput
            filename = workdir+'SbatchSubmit/submit_level2_'+datanum+'_%s.submit'%counter
            f = open(filename, 'w')
            f.write(submit.format(infile=infile,
                                  gcdfile=gcdfile,
                                  hdfoutput=hdfoutput,
                                  outfile=outfile,
                                  datanum=datanum,
                                  particle=particle,
                                  counter=counter,
                                  environment=environment,
                                  useqgsjet=useqgsjet,
                                  script=script))
            submit_file.write('sbatch %s\n'%filename)
            counter += 1
            f.close()
            
submit_file.close()
