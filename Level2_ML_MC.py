#!/usr/bin/env python

'''
This scripts runs level2 scripts (+ few level3 scripts) and prepare data for reconstruction using machine learning.

To Run:
        ./Level2_ML_MC.py 
           -i Level2_IC86.2016_Run128027_IT_0610.i3.bz2
           -o jpt.i3
           -H jpt.h5
           -g GCD.i3

Use /home/rkoirala/icerecdist/TankWise/build/env-shell.sh This has all the necessary Modules and Services.
contact: Ramesh Koirala, rkoirala@udel.edu
'''
import os, sys, glob, numpy
from icecube import icetray, dataclasses, dataio, toprec, recclasses, icetop_Level3_scripts
from icecube import gulliver, lilliput, DomTools, tableio, gulliver_modules, hdfwriter, filterscripts
#from icecube.filterscripts import icetop2station
from icecube.filterscripts.offlineL2.Recalibration import *
from icecube.filterscripts.offlineL2.level2_IceTop_CalibrateAndExtractPulses import *
from icecube.icetop_Level3_scripts import icetop_globals
import icetop2station_level3_16 as icetop2station
from optparse import OptionParser

from weighting import weighting_PolygonatoTG, weighting_H4a, weighting_GST, weighting_GSF

from I3Tray import I3Tray

parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)

parser.add_option("--infile", "-i", help="Input Level1 i3 files")
parser.add_option("--hdfoutput", "-H", help="HDF output file name with path")
parser.add_option("--gcdfile", "-g", help="GCD file")
parser.add_option("--outfile", '-o', help="Output I3 file")
parser.add_option("--particle")
parser.add_option("--isqgsjet", action='store_true', help="QGSJETII-04 interaction model if called.")
opts, args = parser.parse_args()

# Provide Input GCD and i3 files and Output i3 files
inputfiles = glob.glob(opts.infile)
infiles = inputfiles
GCDfile = glob.glob(opts.gcdfile)
hdfout = opts.hdfoutput # Output hdf file where requested keys are booked.
outfile = opts.outfile

name = "ML"

print ''
print opts.infile
print "Input I3 Files: ", infiles
print "GCDFile: ", GCDfile
print "HDF Output: ", hdfout
print ''
print GCDfile + infiles

tray = I3Tray()

HLCPulse='CleanedHLCTankPulses'
SLCPulse='OfflineIceTopSLCTankPulses' 

def removeFiltersNotPassed(frame):
    sta2Passed=False
    if (frame["IceTop_TwoStationFilter_Bool"].value):
        sta2Passed=True
    return sta2Passed

def combine_pulse(frame):
    if (HLCPulse in frame) and (SLCPulse in frame):
        combined_pulse = dataclasses.I3RecoPulseSeriesMap()
        removedSLCQcut = dataclasses.I3RecoPulseSeriesMap()
        cleanhlcpulse = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, HLCPulse)
        offlineslcpulse = frame[SLCPulse]
        for omkey, pulse in cleanhlcpulse:
            if omkey not in combined_pulse:
                combined_pulse[omkey] = pulse
            else:
                pulselist = combined_pulse[omkey]
                for pul in pulse:
                    pulselist.append(pul)
        #for slcomkey, slcpulse in offlineslcpulse:
        #    if slcomkey not in combined_pulse:
        #        combined_pulse[slcomkey]=slcpulse
        #    else:
        #        pulselistslc = combined_pulse[slcomkey]
        #        for pulslc in slcpulse:
        #            pulselistslc.append(pulslc)
        #frame['CombinedHLCSLCPulse'] = combined_pulse
        for slcomkey, slcpulse in offlineslcpulse:
            for pul in slcpulse:
                if pul.charge>0.8:
                    if slcomkey not in combined_pulse:
                        #combined_pulse[slcomkey]=slcpulse
                        combined_pulse[slcomkey]=[pul]
                    else:
                        pulselistslc = combined_pulse[slcomkey]
                        #for pulslc in slcpulse:
                        pulselistslc.append(pul)
        frame['CombinedHLCSLCPulse'] = combined_pulse

def splitPulses(frame):
    '''
    Count number of times a particular station is hit. HLC has two hits on a station and SLC has only one.
    Form a dictionary with station number and number of time its hit.                                     
    If the count is 2 or more put that pulse in HLC, if the number of hit is only one then put that pulse in
    SLC pulse.
    '''
    hlc_pulses = dataclasses.I3RecoPulseSeriesMap()
    slc_pulses = dataclasses.I3RecoPulseSeriesMap()

    seededRTpulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'IT_RT_180m_450ns')
    stations     = {}
    keepomkeys   = []
    removeomkeys = []

    for omkey, pulse in seededRTpulses:
        if omkey.string not in stations:
            stations[omkey.string] = 1
        else:
            stations[omkey.string] += 1

    for omkey, pulse in seededRTpulses:
        if stations[omkey.string] >= 2: # HLC has hit on two tanks of a station.
            hlc_pulses[omkey] = pulse
        else:
            slc_pulses[omkey] = pulse   # SLC has hit on one tank of a station.

    # Now save them in a container.
    frame["SeededRTHLCPulses"] = hlc_pulses
    frame["SeededRTSLCPulses"] = slc_pulses
"""
def splitPulses(frame):
    '''
    Count number of times a particular station is hit. HLC has two hits on a station and SLC has only one.
    Form a dictionary with station number and number of time its hit.                                     
    If the count is 2 or more put that pulse in HLC, if the number of hit is only one then put that pulse in
    SLC pulse.
    '''
    hlc_pulses = dataclasses.I3RecoPulseSeriesMap()
    slc_pulses = dataclasses.I3RecoPulseSeriesMap()
    rm_pulses  = dataclasses.I3RecoPulseSeriesMap()
    
    seededRTpulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'IT_RT_180m_450ns')
    stations     = {}
    keepomkeys   = []
    removeomkeys = []

    # separate omkeys based on which one to keep and which one to remove.
    for omkey, pulse in seededRTpulses:
        for pul in pulse:
            if (pul.charge>0.1):
                keepomkeys.append(omkey)
            else:
                removeomkeys.append(omkey)
               
    # From keepkeys, find out how many of them hits 2 (hlc) or 1 (slc) tanks. 
    for omkey in keepomkeys:
        if omkey.string not in stations:
            stations[omkey.string] = 1
        else:
            stations[omkey.string] += 1
            
    # Separate HLC, SLC and removed keys.
    for omkey, pulse in seededRTpulses:
        if omkey in keepomkeys:
            if stations[omkey.string] >= 2: # HLC has hit on two tanks of a station.
                hlc_pulses[omkey] = pulse
            elif stations[omkey.string] == 1:
                slc_pulses[omkey] = pulse   # SLC has hit on one tank of a station.
        elif omkey in removeomkeys:
            rm_pulses[omkey] = pulse

    # Now save them in a container.
    frame["SeededRTHLCPulses"] = hlc_pulses
    frame["SeededRTSLCPulses"] = slc_pulses
    frame["rmThresholdPulses"] = rm_pulses
"""
# Read input files frame by frame
tray.Add("I3Reader","reader")(
         ("FileNameList", GCDfile + infiles)
         )

# Input  = IceTopRawData
# Output = CleanIceTopRawData
tray.Add('I3DOMLaunchCleaning', 'DOMLaunchCleaning',
         CleanedKeysList = 'BadDomsListSLC'
         )

# http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/filterscripts/trunk/python/offlineL2/Recalibration.py
# Input  = CleanIceTopRawData
# IceTopCalibration does
#  a. "I3WaveCalibrator"
#  b. "I3WaveformSplitter"
#  c. "I3TopHLCPulseExtractor"
#  d. "I3TopSLCPulseExtractor"
# Output = ReextractedIceTopPulses and ReextractedIceTopPulses_SLC
tray.Add(IceTopCalibration, 'IceTopCalibration')

def dstofflinemerge(frame, Output='', Input=[]):
    rootpulses = []
    for i in Input:
        if not i in frame:
            continue
        rootpulses.append(i)
    frame[Output] = dataclasses.I3RecoPulseSeriesMapUnion(frame, rootpulses)

# Unify pulses after using "I3TopHLCPulseExtractor" and "I3TopSLCPulseExtractor"
# 'IceTopDSTOnlyPulses' is not in the frame, but kept here for future use if needed.
tray.Add(dstofflinemerge, 'dstofflinemerge',
         Input   = ['IceTopDSTOnlyPulses',
                    'ReextractedIceTopPulses',
                    'ReextractedIceTopPulses_SLC'],
         Output  = 'IceTopPulses',
         Streams = [icetray.I3Frame.DAQ]
         )

# http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/filterscripts/trunk/python/offlineL2/level2_IceTop_CalibrateAndExtractPulses.py     
# Input  = IceTopPulses
#      IceTopPulses is unification of SLC and HLC pe pulse. Later SLC will be ignored down the processing line.
# CalibrateAndExtractIceTop does:
#  a. "I3VEMConverter" : convert pe to VEM
#  b. "IceTopWaveformSplitter" : No clue what this does. May repeat IceTopCalibration???
#  c. "I3HLCTankPulseMerger"
#  d. "I3TopHLCClusterCleaning"
#  e. "DistributePnFObjects" : Distribute pole frame objects that were from the IceTopSplit??
# Output = "CleanedHLCTankPulses"
tray.Add(CalibrateAndExtractIceTop, 'CalibrateAndExtractIceTop',
         Pulses = 'IceTopPulses',
         IceTopPhysicsStream = 'ice_top', # default is 'IceTopSplit'
         )

# Run STA2 filter check on P frames.                                                                                                                          
# This module does not remove any frames.                                                                                                                     
# Input Pulse: icetop_globals.CleanedHLCTankPulses                                                                                                            
# Output: "IceTop_TwoStationFilter_Bool"                                                                                                                      
tray.AddSegment(icetop2station.IceTopTwoStationFilter, "STA2Filter") # Written by Timo.                                                                       

tray.AddModule(removeFiltersNotPassed,"keepSTA2only")

# ----------------------------------------------------------------------------------------------------------------
# This is required to run background simulator below.
# See L79: projects/icetop_Level3_scripts/python/segments/level3_IceTop.py 
tray.AddModule(lambda frame: frame.Stop != icetray.I3Frame.Physics, name + "_physics_dropper")

from icecube.tpx.segments import CalibrateSLCs
tray.AddSegment(CalibrateSLCs, name+'_OfflineIceTopSLCVEMPulses',
                SLCVEMPulses=icetop_globals.icetop_slc_vem_pulses,   # input
                SLCTankPulses=icetop_globals.icetop_slc_pulses,       # output, together with icetop_globals.icetop_slc_vem_pulses+'Calibrated' and TankPulseMergerExcludedSLCTanks
                )

# SLC time correction
tray.AddModule(icetop_Level3_scripts.modules.I3IceTopSLCTimeCorrect,name+'_SLCTimeCorrect',
               SLCPulses=icetop_globals.icetop_slc_pulses,
               If=lambda fr: icetop_globals.icetop_slc_pulses in fr and len(fr[icetop_globals.icetop_slc_pulses])>0)

# Note: RecalibrateVEMPulses is NOT done for MC.

# http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/icetop_Level3_scripts/trunk/python/segments/SimulateBackground.py
# http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/hdembinski/top-background-simulator/private/top-background-simulator/TopBackgroundSimulator.cxx
# Note: HLCClusterCleaning is run again on background mixed pulses to produce CleanedHLCTankPulses.
# Input:  'OfflineIceTopHLCTankPulses'
# Output: 'CleanedHLCTankPulses'

# Note: This does not run with TankWise icerec.
# Do background simulation after slc calibration.
tray.AddSegment(icetop_Level3_scripts.segments.SimulateBackground, name + '_ITbackground',
                #HLCTankPulses=HLCPulse, # CleanedHLCTankPulses does not work. Output will be new CleanedHLCTankPulses.
                HLCTankPulses="OfflineIceTopHLCTankPulses",
                SLCTankPulses=SLCPulse,
                NoiseRate=1500, #Default=1500
                AddJitter=False)
                

# Preparing for SRT Cleaning. Use HLC as seed and clean combined pulses of HLC and SLC.
tray.Add(combine_pulse, 'combine_hlc_and_slc')

# Read https://wiki.icecube.wisc.edu/index.php/SLC_hit_cleaning
# http://code.icecube.wisc.edu/svn/projects/icetop_Level3_scripts/trunk/python/segments/CleanIceTop.py
# SeededRTCleaning is done here.
tray.AddSegment(icetop_Level3_scripts.segments.CleanIceTop,name+'_clean_it',
                detect_conf     = 'IC86.2016',
                it_pulses       = 'CombinedHLCSLCPulse',
                it_pulses_clean = 'IT_RT_180m_450ns',
                excluded_tanks  = 'ClusterCleaningExcludedTanks',
                extra_excluded_tanks='ExtraCleanedExcludedTanks'
                )

# Input: 'IT_RT_180m_450ns'
# Output: 'SeededRTHLCPulses', 'SeededRTSLCPulses'
tray.Add(splitPulses, 'split_seededRT_pulses')

# Rerun filter check here using the cleanest pulses. 
# Produces FilterMask, IceTop_TwoStationFilter, IceTop_InFillFilter, and IceTop_StandardFilter.
# /IceCube/projects/icetop_Level3_scripts/branches/sta2events/python/segments/ReRunFilters.py
tray.AddSegment(icetop_Level3_scripts.segments.ReRunFilters,name+"_rerunFilters",
                Detector="IC86.2016",
                isMC=False,
                Pulses='SeededRTHLCPulses'
               ) 

seededRTHLCPulse='SeededRTHLCPulses'
seededRTSLCPulse='SeededRTSLCPulses'
excluded        ='ExtraCleanedExcludedTanks'

def Nstation_Nslc_Qtotal(frame):
    if seededRTHLCPulse in frame:
        nsta       = dataclasses.I3Double()
        nslc       = dataclasses.I3Double()
        ntanks     = dataclasses.I3Double()
        qtotal     = dataclasses.I3Double()
        qtotalh    = dataclasses.I3Double()
        qtothlcslc = dataclasses.I3Double()
        qmax       = dataclasses.I3Double()
        loudsta    = dataclasses.I3Double()
        weight     = dataclasses.I3Double()
        weight_all = dataclasses.I3Double()
        weight_h4a      = dataclasses.I3Double()
        weight_all_h4a  = dataclasses.I3Double()
        weight_gst      = dataclasses.I3Double()
        weight_all_gst  = dataclasses.I3Double()
        weightL_gsf     = dataclasses.I3Double()
        weight_allL_gsf = dataclasses.I3Double()
        weightM_gsf     = dataclasses.I3Double()
        weight_allM_gsf = dataclasses.I3Double()
        weightH_gsf     = dataclasses.I3Double()
        weight_allH_gsf = dataclasses.I3Double()
        hlcpulses       = frame[seededRTHLCPulse]
        slcpulses       = frame[seededRTSLCPulse]
        srthlcslcpulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, "IT_RT_180m_450ns")
        mcprimary       = frame['MCPrimary']
        weight.value    = weighting_PolygonatoTG(numpy.array([mcprimary.energy]), opts.particle, isqgsjet=opts.isqgsjet)
        weight_all.value     = weighting_PolygonatoTG(numpy.array([mcprimary.energy]), opts.particle, 
                                                do='all_particle_flux', isqgsjet=opts.isqgsjet)
        weight_h4a.value     = weighting_H4a(numpy.array([mcprimary.energy]), opts.particle, isqgsjet=opts.isqgsjet)
        weight_all_h4a.value = weighting_H4a(numpy.array([mcprimary.energy]), opts.particle, 
                                                do='all_particle_flux', isqgsjet=opts.isqgsjet)
        weight_gst.value     = weighting_GST(numpy.array([mcprimary.energy]), opts.particle, isqgsjet=opts.isqgsjet)
        weight_all_gst.value = weighting_GST(numpy.array([mcprimary.energy]), opts.particle, 
                                                do='all_particle_flux', isqgsjet=opts.isqgsjet)
        '''
        # Trouble doing this because numpy.linalg.multi_dot does not exist in cvmfs numpy.
        weightL_gsf.value    = weighting_GSF(numpy.array([mcprimary.energy]), opts.particle, which_flux='low')
        weight_allL_gsf.value= weighting_GSF(numpy.array([mcprimary.energy]), opts.particle, which_flux='low', 
                                    do='all_particle_flux')
        weightM_gsf.value    = weighting_GSF(numpy.array([mcprimary.energy]), opts.particle, which_flux='median')
        weight_allM_gsf.value= weighting_GSF(numpy.array([mcprimary.energy]), opts.particle, which_flux='median', 
                                    do='all_particle_flux')
        weightH_gsf.value    = weighting_GSF(numpy.array([mcprimary.energy]), opts.particle, which_flux='high')
        weight_allH_gsf.value= weighting_GSF(numpy.array([mcprimary.energy]), opts.particle, which_flux='high', 
                                    do='all_particle_flux')
        '''
        nsta.value           = len(set([omkey.string for omkey, pulse in hlcpulses]))
        nslc.value           = len([omkey.string for omkey, pulse in slcpulses])
        ntanks.value         = len([omkey.string for omkey, pulse in srthlcslcpulses])
        QlistHLC             = [pul.charge for omkey, pulse in hlcpulses for pul in pulse]
        QlistTank            = sorted([pul.charge for omkey, pulse in srthlcslcpulses for pul in pulse])
        charge_sta           = [(pul.charge,omkey.string) for omkey, pulse in hlcpulses for pul in pulse]
        loudsta.value        = max(charge_sta)[1] in [26,36,46,79,80,81]
        qtotal.value         = numpy.sum(QlistHLC)
        qtotalh.value        = numpy.sum(QlistTank[:-1])
        qtothlcslc.value     = numpy.sum(QlistTank)
        qmax.value           = numpy.max(QlistTank)
        frame['Nslc']              = nslc
        frame['Nstation']          = nsta
        frame['Ntanks']            = ntanks
        frame['TotalChargeHLC']    = qtotal
        frame['TotalChargeSum']    = qtothlcslc
        frame['TotalChargeHillas'] = qtotalh
        frame['Qmax']              = qmax
        frame['LoudestStation']    = loudsta
        frame['WeightH4a']         = weight_h4a
        frame['WeightGST']         = weight_gst
        frame['WeightPolygonatoTG']= weight
        frame['Weight1ParticleAssumptionPolygonatoTG']     = weight_all
        frame['Weight1ParticleAssumptionH4a']              = weight_all_h4a
        frame['Weight1ParticleAssumptionGST']              = weight_all_gst
        '''
        frame['WeightGSFL']  = weightL_gsf
        frame['WeightGSFM']  = weightM_gsf
        frame['WeightGSFH']  = weightH_gsf
        frame['Weight1ParticleAssumptionGSFL']  = weight_allL_gsf
        frame['Weight1ParticleAssumptionGSFM']  = weight_allM_gsf
        frame['Weight1ParticleAssumptionGSFH']  = weight_allH_gsf
        '''
tray.Add(Nstation_Nslc_Qtotal, 'nsta_nslc')

# ------------------------------------------------------------
# TankWiseLaputop Reconstruction is not done here.
# Machine learning approach is used later for reconstruction.
# ------------------------------------------------------------
# Seed "ShowerCOG" and "ShowerPlane" already exist on the frame.
tray.AddModule( "I3TopRecoCore", "Core" ) (
    ( "datareadout",       seededRTHLCPulse),      # ! Use first IT event pulses
    ( "showercore",        "ShowerCOG" ),  # Default
    ( "weighting_power",   0.5 ),          # Default
    ( "verbose",           False )         # Default
    #("ntanks",             4 )            # Default = -1 (use all tanks)
    )

# Reconstruct the shower plane
# Code: ~/icerecdist/src.trunk/toprec/private/toprec/I3TopRecoPlane.cxx 
tray.AddModule( "I3TopRecoPlane", "Plane" ) (
    ( "EventHeaderName",   "I3EventHeader" ),      # Default
    ( "DataReadout",       seededRTHLCPulse),      # ! Use first IT event pulses
    ( "ShowerPlane",       "ShowerPlane" ),        # Default
    ( "Trigger",           2 ),                    # Default
    ( "Verbose",           False )                 # Default
    )

hdf = hdfwriter.I3HDFTableService(hdfout)
tray.AddModule(tableio.I3TableWriter, 'TableWriter',
               TableService    =[hdf],
               SubEventStreams = ['ice_top', 'ice_top2'], # 'IceTopSplit' for exp data.
               Keys            = ['I3EventHeader', 'MCPrimary',
                                  "ShowerCOG", "ShowerPlane",
                                  'IT_RT_180m_450ns', "rmThresholdPulses",
                                  "OfflineIceTopSLCVEMPulses",
                                  "SeededRTHLCPulses", "SeededRTSLCPulses",
                                  "IceTop_StandardFilter", 'IceTopSTA5_13',
                                  "IceTop_InFillFilter", "IceTop_TwoStationFilter",
                                  'Nstation', 'Nslc', 'Ntanks',
                                  'TotalChargeHLC', 'TotalChargeHillas', 'TotalChargeSum', 'Qmax',
                                  'LoudestStation',
                                  'WeightPolygonatoTG', 'Weight1ParticleAssumptionPolygonatoTG',
                                  'WeightH4a', 'Weight1ParticleAssumptionH4a',
                                  'WeightGST', 'Weight1ParticleAssumptionGST'
                                  ])

'''
tray.AddModule("I3Writer", "EventWriter")(
    ("Streams", [icetray.I3Frame.DAQ, icetray.I3Frame.Physics]),
    ("DropOrphanStreams", [icetray.I3Frame.DAQ]),
    ("Filename", outfile))

'''
tray.Execute()

tray.Finish()
