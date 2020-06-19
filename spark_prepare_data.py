#!/usr/bin/env python

"""
This code is required to produce TankX, TankY, TankZ, HitsTime, and Pulses which are of shape
(number of events, 35). These variables are required to predict X, Y, zenith and energy 
for experimental data.
"""

import tables, numpy, glob
import sys, os
from optparse import OptionParser

parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
parser.add_option('-f', "--filename", help="Input hdf file for each run.")
opts, args = parser.parse_args()

thisdir  = '/data/icet0/rkoirala/LowEnergy/RandomForest/'
hdffile  = opts.filename

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
StaHits   = {26:0, 36:0, 46:0, 79:0, 80:0, 81:0}
leq      = 35

print 'Preparing data for Predicting X, Y, Zenith and Energy.'
#datadir="/data/icet0/rkoirala/DATA/Experiment/2016/2016ML/"

# Start collecting all the necessary information.
hf = tables.open_file(hdffile, 'a')
if ('SeededRTSLCPulses' in hf.root):

    Pulses   = []
    HitsTime = []
    TankX    = []
    TankY    = []
    TankZ    = []

    print hdffile
    
    # Pulses , IT_RT_180m_450ns, SeededRTHLCPulses
    pulses       = hf.root.SeededRTHLCPulses.cols.charge[:]
    string       = hf.root.SeededRTHLCPulses.cols.string[:]
    om           = hf.root.SeededRTHLCPulses.cols.om[:]
    time         = hf.root.SeededRTHLCPulses.cols.time[:]
    start        = hf.root.__I3Index__.SeededRTHLCPulses.cols.start[:]
    stop         = hf.root.__I3Index__.SeededRTHLCPulses.cols.stop[:]

    # ==================================================================
    start_ = start#[basicmask]
    stop_  = stop#[basicmask]

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
        
        for sta in set(stations):
            if (sta==26 or sta==36 or sta==46 or sta==79 or sta==80 or sta==81):
                StaHits[sta]+= 1

    if 'TankX' in hf.root:
        hf.remove_node('/', 'TankX')
    if 'TankY' in hf.root:
        hf.remove_node('/', 'TankY')
    if 'TankZ' in hf.root:
        hf.remove_node('/', 'TankZ')
    if 'Pulses' in hf.root:
        hf.remove_node('/', 'Pulses')
    if 'HitsTime' in hf.root:
        hf.remove_node('/', 'HitsTime')
    if 'numStationHit' in hf.root:
        hf.remove_node('/', 'numStationHit')
    hf.create_array('/', 'TankX', numpy.array(TankX))
    hf.create_array('/', 'TankY', numpy.array(TankY))
    hf.create_array('/', 'TankZ', numpy.array(TankZ))
    hf.create_array('/', 'Pulses', numpy.array(Pulses))
    hf.create_array('/', 'HitsTime', numpy.array(HitsTime))

    class Particle(tables.IsDescription):
        sta26        = tables.Float32Col()
        sta36        = tables.Float32Col()
        sta46        = tables.Float32Col()                                                                              
        sta79        = tables.Float32Col()
        sta80        = tables.Float32Col()
        sta81        = tables.Float32Col()

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
                
else:
    hf.close()
