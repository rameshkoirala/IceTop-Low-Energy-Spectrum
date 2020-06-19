#!/usr/bin/env python

'''
Save data in a single file for easier and faster processing.

Simulation: 
    Collect data from all particles (proton, helium, oxygen, and iron) and all energy bins (log10GeV: 4-7.4) 
    and combine them into a single file.
    This file (one for Sibyll2.1 and one for QGSJetII-04) will be used for further processing 
    of simulation. No need to come back to the Level2 folder again and again.

Experiment:
    Collect data from Runs ending on '0'. This file will be used for data-mc comparison.

To Run:
    $ python booking_data_mc.py --isMC
    $ python booking_data_mc.py --isMC --isqgsjet
    $ python booking_data_mc.py --isExp
'''

import tables, numpy, glob
import sys, os
from optparse import OptionParser

parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)

parser.add_option("--isqgsjet", action='store_true', help="QGSJETII-04 interaction model if called.")
parser.add_option("--isMC", action='store_true', help="book level2 data for simulation.")
parser.add_option("--isExp", action='store_true', help="book level2 data for experiment.")

opts, args = parser.parse_args()

thisdir  = '/data/icet0/rkoirala/LowEnergy/RandomForest/'

# Required to get hits tank position, its charge, and its time.
import pickle
om_dict       = {61:'A', 62:'A', 63:'B', 64:'B'}
tankpos_file  = open(thisdir+'icetop_geometry.txt', 'rb')
tank_position = pickle.load(tankpos_file)
tankpos_file.close()

tpx = numpy.array([j[0] for i in tank_position.values() for j in i.values()])
tpy = numpy.array([j[1] for i in tank_position.values() for j in i.values()])
tpz = numpy.array([j[2] for i in tank_position.values() for j in i.values()])

# This part provides the index where information of each tank of IceTop is stored.
index_dict = {}
count = 0
for sta in range(1,82):
    index_dict[sta] = {'A':count, 'B':count+1}
    count += 2

# Append data from each file in these array.
Qmax      = numpy.array([])
Qhillas   = numpy.array([])
Qtotal    = numpy.array([])
Qtotalhlc = numpy.array([])
Nsta      = numpy.array([])
Nslc      = numpy.array([])
Ntanks    = numpy.array([])
Plane_Zenith  = numpy.array([])
Plane_Azimuth = numpy.array([])
FiltCondt = numpy.array([])
LoudSta   = numpy.array([])
COGX      = numpy.array([])
COGY      = numpy.array([])
T0        = numpy.array([])
PredictedX= numpy.array([])
PredictedY= numpy.array([])
PredictedZen           = numpy.array([])
PredictedLogEnergy_h4a = numpy.array([])
StaHits   = {26:0, 36:0, 46:0, 79:0, 80:0, 81:0}

Pulses   = []
HitsTime = []
TankX    = []
TankY    = []
TankZ    = []
sum_wt   = 0
counter  = 0
leq      = 35

if opts.isMC:
    print 'Booking MC data.'
    datadir  = '/data/icet0/rkoirala/DATA/Simulation/*/*/Level2/'

    if opts.isqgsjet:
        final_hdf_filename = thisdir+'analysis_simulation_HLCCoreSeed_qgsjet_all.h5'
        allfiles = glob.glob(datadir+'Level2*Background_SRT_ML_isqgsjet*.h5')
    else:
        final_hdf_filename = thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_all.h5'
        allfiles = glob.glob(datadir+'Level2*Background_SRT_HLCCoreSeed_slcQcut_ML*.h5')

    # Set up empty array to fill it later.
    PDG       = numpy.array([])
    Tcorex    = numpy.array([])
    Tcorey    = numpy.array([])
    Rcorex    = numpy.array([])
    Rcorey    = numpy.array([])
    Rzenith   = numpy.array([])
    Razimuth  = numpy.array([])
    Tzenith   = numpy.array([])
    Tazimuth  = numpy.array([])
    Energy    = numpy.array([])
    WeightPoly= numpy.array([])
    WeightH4a = numpy.array([])
    WeightGST = numpy.array([])
    WeightL   = numpy.array([])
    WeightM   = numpy.array([])
    WeightH   = numpy.array([])
    
elif opts.isExp:
    print 'Booking Experiment data.'
    datadir="/data/icet0/rkoirala/DATA/Experiment/2016/2016ML/"
    
    allfiles = glob.glob(datadir+'Run128*0_ML_slcQcut.h5')
    final_hdf_filename = thisdir+'analysis_experiment_HLCCoreSeed_slcQcut_0.h5'
else:
    raise Exception("Provide one of the following option: --isMC or --isExp")

print "Files to process: %i"%len(allfiles)

duration = 0
# Start collecting all the necessary information.
for hdffile in allfiles: # file_list defined in steering file.
    hf = tables.open_file(hdffile)
    if ('SeededRTSLCPulses' in hf.root):
        if opts.isMC and counter%100==0:
            print hdffile, "....", counter
        if opts.isExp:
            print hdffile, "....", counter
        counter += 1
        filtercondition  = hf.root.IceTop_TwoStationFilter.cols.value[:]
        cog_x        = hf.root.ShowerCOG.cols.x[:]
        cog_y        = hf.root.ShowerCOG.cols.y[:]
        # Angle
        plane_zenith = hf.root.ShowerPlane.cols.zenith[:]
        plane_azimuth= hf.root.ShowerPlane.cols.azimuth[:]
        t0           = hf.root.ShowerCOG.cols.time[:]
        # Charge Info
        qmax         = hf.root.Qmax.cols.value[:]
        qhillas      = hf.root.TotalChargeHillas.cols.value[:]
        qtotalhlc    = hf.root.TotalChargeHLC.cols.value[:]
        qtotal       = hf.root.TotalChargeSum.cols.value[:]
        # Tank-Station hit
        nsta         = hf.root.Nstation.cols.value[:]
        nslc         = hf.root.Nslc.cols.value[:]
        ntanks       = hf.root.Ntanks.cols.value[:]
        # Rest
        loudsta      = hf.root.LoudestStation.cols.value[:]
        dur          = hf.root.duration[:]
        # Prediction
        predictedX   = hf.root.PredictedX[:]
        predictedY   = hf.root.PredictedY[:]
        predictedZen = hf.root.PredictedZen[:]
        predictedLogE= hf.root.PredictedLogEnergy_h4a[:]
        # Pulses , IT_RT_180m_450ns, SeededRTHLCPulses
        pulses       = hf.root.SeededRTHLCPulses.cols.charge[:]
        string       = hf.root.SeededRTHLCPulses.cols.string[:]
        om           = hf.root.SeededRTHLCPulses.cols.om[:]
        time         = hf.root.SeededRTHLCPulses.cols.time[:]
        start        = hf.root.__I3Index__.SeededRTHLCPulses.cols.start[:]
        stop         = hf.root.__I3Index__.SeededRTHLCPulses.cols.stop[:]
        if opts.isMC:
            energy       = hf.root.MCPrimary.cols.energy[:]
            pdg_encoding = hf.root.MCPrimary.cols.pdg_encoding[:]
            true_x       = hf.root.MCPrimary.cols.x[:]
            true_y       = hf.root.MCPrimary.cols.y[:]
            true_z       = hf.root.MCPrimary.cols.z[:]
            true_zenith  =  hf.root.MCPrimary.cols.zenith[:]
            true_azimuth = hf.root.MCPrimary.cols.azimuth[:]
                
            weightpoly = hf.root.WeightPolygonatoTG.cols.value[:]
            weighth4a  = hf.root.WeightH4a.cols.value[:]
            weightgst  = hf.root.WeightGST.cols.value[:]
        hf.close()
        
        # ==================================================================
        basicmask = (filtercondition==1)

        start_ = start[basicmask]
        stop_  = stop[basicmask]

        for i in range(len(start_)):
            tmp_pul = numpy.zeros(leq)
            tmp_x   = numpy.zeros(leq)
            tmp_y   = numpy.zeros(leq)
            tank_x  = numpy.zeros(leq)
            tank_y  = numpy.zeros(leq)
            tank_z  = numpy.zeros(leq)
            time_   = numpy.zeros(leq)
            stations= string[start_[i]:stop_[i]] # stations list for one event.                                                                 
            tanks   = om[start_[i]:stop_[i]]
            if len(stations)>0:
                index   = numpy.array([index_dict[stations[j]][om_dict[tanks[j]]] for j in range(len(stations))])

                evt_pulse = pulses[start_[i]:stop_[i]]
                evt_x     = tpx[index]
                evt_y     = tpy[index]
                evt_z     = tpz[index]
                evt_time  = time[start_[i]:stop_[i]]
                        
                npcolstack       = numpy.column_stack((evt_pulse, evt_x, evt_y, evt_z)) # sort with highest charge first.
                sortedcolstack   = npcolstack[(npcolstack[:,0].argsort())[::-1]] # sorting with highest charge first.
                lindex = len(index)
                if lindex>leq:
                    lindex = leq
                tmp_pul[:lindex] = sortedcolstack[:lindex,0]
                tank_x[:lindex]  = sortedcolstack[:lindex,1]  # indexed from 1 to 162
                tank_y[:lindex]  = sortedcolstack[:lindex,2]  # indexed from 1 to 162
                tank_z[:lindex]  = sortedcolstack[:lindex,3]
            
                #sorted_evt_time  = numpy.diff(numpy.array(sorted(evt_time)))
                sorted_evt_time  = numpy.array(sorted(evt_time)) - min(evt_time)
                time_[:lindex-1] = sorted_evt_time[:lindex-1]
                time_[lindex-1:] = numpy.array((leq-(lindex-1))*[0])
            
            Pulses.append(tmp_pul)
            HitsTime.append(time_)
            TankX.append(tank_x)
            TankY.append(tank_y)
            TankZ.append(tank_z)
            
            if opts.isMC:
                wt = weighth4a[i] # rate of station hit
            elif opts.isExp:
                wt = 1            # number of station hit
                
            for sta in set(stations):
                if (sta==26 or sta==36 or sta==46 or sta==79 or sta==80 or sta==81):
                    StaHits[sta]+= wt
                    
        # ====================================================================== #
        FiltCondt = numpy.append(FiltCondt, filtercondition[basicmask])
        LoudSta   = numpy.append(LoudSta  , loudsta[basicmask])
        Qmax      = numpy.append(Qmax     , qmax[basicmask])
        Qhillas   = numpy.append(Qhillas  , qhillas[basicmask])
        Qtotal    = numpy.append(Qtotal   , qtotal[basicmask])
        Qtotalhlc = numpy.append(Qtotalhlc, qtotalhlc[basicmask])
        Nsta      = numpy.append(Nsta     , nsta[basicmask])
        Nslc      = numpy.append(Nslc     , nslc[basicmask])
        Ntanks    = numpy.append(Ntanks   , ntanks[basicmask])
        COGX      = numpy.append(COGX     , cog_x[basicmask])
        COGY      = numpy.append(COGY     , cog_y[basicmask])
        T0        = numpy.append(T0       , t0[basicmask])
        Plane_Zenith  = numpy.append(Plane_Zenith, plane_zenith[basicmask])
        Plane_Azimuth = numpy.append(Plane_Azimuth, plane_azimuth[basicmask])
        PredictedX    = numpy.append(PredictedX, predictedX[basicmask])
        PredictedY    = numpy.append(PredictedY, predictedY[basicmask])
        PredictedZen  = numpy.append(PredictedZen, predictedZen[basicmask])
        PredictedLogEnergy_h4a = numpy.append(PredictedLogEnergy_h4a, predictedLogE[basicmask])
        duration += dur
        if opts.isMC:
            Tcorex    = numpy.append(Tcorex   , true_x[basicmask])
            Tcorey    = numpy.append(Tcorey   , true_y[basicmask])
            Tzenith   = numpy.append(Tzenith  , true_zenith[basicmask])
            Tazimuth  = numpy.append(Tazimuth , true_azimuth[basicmask])
            Energy    = numpy.append(Energy   , energy[basicmask])
            WeightPoly= numpy.append(WeightPoly, weightpoly[basicmask])
            WeightH4a = numpy.append(WeightH4a , weighth4a[basicmask])
            WeightGST = numpy.append(WeightGST , weightgst[basicmask])
            PDG       = numpy.append(PDG       , pdg_encoding[basicmask])
    else:
        hf.close()

class Particle(tables.IsDescription):
    sta26        = tables.Float32Col()
    sta36        = tables.Float32Col()
    sta46        = tables.Float32Col()                                                                              
    sta79        = tables.Float32Col()
    sta80        = tables.Float32Col()
    sta81        = tables.Float32Col()

# Now save all the resulted array in a .h5 file so that we don't have to run 
#  this script again and again. Running this script from the beginning can take
#  upto 5 mins.
hf = tables.open_file(final_hdf_filename, 'w')
# For both data and MC
hf.create_array('/', 'Qmax', Qmax)
hf.create_array('/', 'Qhillas', Qhillas)
hf.create_array('/', 'Qtotal', Qtotal)
hf.create_array('/', 'Qtotalhlc', Qtotalhlc)
hf.create_array('/', 'Nsta', Nsta)
hf.create_array('/', 'Nslc', Nslc)
hf.create_array('/', 'Ntanks', Ntanks)
hf.create_array('/', 'COGX', COGX)
hf.create_array('/', 'COGY', COGY)
hf.create_array('/', 'T0', T0)
hf.create_array('/', 'PlaneZenith', Plane_Zenith)
hf.create_array('/', 'PlaneAzimuth', Plane_Azimuth)
hf.create_array('/', 'FiltCondt', FiltCondt)
hf.create_array('/', 'LoudSta', LoudSta)
hf.create_array('/', 'TankX', numpy.array(TankX))
hf.create_array('/', 'TankY', numpy.array(TankY))
hf.create_array('/', 'TankZ', numpy.array(TankZ))
hf.create_array('/', 'Pulses', numpy.array(Pulses))
hf.create_array('/', 'HitsTime', numpy.array(HitsTime))
hf.create_array('/', 'PredictedX', PredictedX)
hf.create_array('/', 'PredictedY', PredictedY)
hf.create_array('/', 'PredictedZen', PredictedZen)
hf.create_array('/', 'PredictedLogEnergy_h4a', PredictedLogEnergy_h4a)
hf.create_array('/', 'duration', duration)

if opts.isMC:
    hf.create_array('/', 'pdg_encoding', PDG)
    hf.create_array('/', 'Tcorex', Tcorex)
    hf.create_array('/', 'Tcorey', Tcorey)
    hf.create_array('/', 'Tzenith', Tzenith)
    hf.create_array('/', 'Tazimuth', Tazimuth)
    hf.create_array('/', 'Energy', Energy)
    hf.create_array('/', 'WeightPoly', WeightPoly)
    hf.create_array('/', 'WeightH4a', WeightH4a)
    hf.create_array('/', 'WeightGST', WeightGST)

table = hf.create_table('/', 'numStationHit', Particle)
row   = table.row
row['sta26'] = float(StaHits[26])
row['sta36'] = float(StaHits[36])
row['sta46'] = float(StaHits[46])
row['sta79'] = float(StaHits[79])
row['sta80'] = float(StaHits[80])
row['sta81'] = float(StaHits[81])
row.append()
table.flush()

hf.close()

print "Done Appending and saved in", final_hdf_filename
