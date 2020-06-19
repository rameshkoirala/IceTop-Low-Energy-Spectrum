#!/bin/bash

unzipped="/data/user/rkoirala/DATA/"$5"/"$3"/"$2
outfile="/data/user/rkoirala/DATA/"$5"/"$3"/DetectorSim/"$2"_Oct2016Snow.i3.bz2"
asterixfile="/data/icet0/rkoirala/DATA/Simulation/"$5"/"$3"/DetectorSim/"$2"_Oct2016Snow.i3.bz2"

echo "Unzipped Input: " $unzipped
echo "Output:" $outfile
echo "Enebin:" $3
echo "DataSet:" $5
echo "Runnum: " $4
echo "Resampling Radius: " $6
echo "Resampling Times: " $7
echo "Observation Level: " $8
echo "================================="

bunzip2 -c $1 > $unzipped

python /data/user/rkoirala/CORSIKA/fullSimulation.py --resampling_radius $6 --raise_observation $8 --gcd /data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2016.57531_V0_OctSnow.i3.gz --generator corsika-icetop --with-icetop --detector IC86 -n $7 --without-photon-prop --top-response g4 -r $4 -o $outfile $unzipped

rm $unzipped

rsync --remove-source-files -av $outfile rkoirala@asterix.bartol.udel.edu:$asterixfile