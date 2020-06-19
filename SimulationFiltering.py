#!/usr/bin/env python

### The simulation L1 processing script ###

"""
Icerec Environment: /home/rkoirala/icerecdist/V05-00-05/build/env-shell.sh
To Run:
    ./SimulationFiltering.py -o <output> -i <inputfile> -g <GCDfile> --sdst-archive
"""

from I3Tray import *
from icecube import icetray, dataclasses, dataio, filter_tools, trigger_sim
from icecube import phys_services
from icecube.filterscripts import filter_globals
from icecube.filterscripts.all_filters import OnlineFilter
from icecube.phys_services.which_split import which_split
import os, sys, time, glob
import subprocess
from math import log10, cos, radians
from optparse import OptionParser
from os.path import expandvars

def make_parser():
    """Make the argument parser"""
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--input", 
                      action="store",
                      type="string", 
                      default="", 
                      dest="infile",
                      help="Input i3 file(s)  (use comma separated list for multiple files)")
    parser.add_option("-g", "--gcd", 
                      action="store",
                      type="string", 
                      default="", 
                      dest="gcdfile",
                      help="GCD file for input i3 file")
    parser.add_option("-o", "--output", 
                      action="store",
                      type="string", 
                      default="", 
                      dest="outfile",
                      help="Output i3 file")
    parser.add_option("-n", "--num", 
                      action="store",
                      type="int", 
                      default=-1, 
                      dest="num",
                      help="Number of frames to process")
    parser.add_option("--qify", 
                      action="store_true",
                      default=False, 
                      dest="qify",
                      help="Apply QConverter, use if file is P frame only")
    parser.add_option("--MinBiasPrescale", 
                      action="store",
                      type="int", 
                      default=None, 
                      dest="MinBiasPrescale",
                      help="Set the Min Bias prescale to something other than default")
    parser.add_option("--photonicsdir", 
                      action="store",
                      type="string", 
                      default="/data/sim/sim-new/PhotonTablesProduction",
                      dest="photonicsdir", 
                      help="Directory with photonics tables")
    parser.add_option("--sdst-archive", 
                      action="store_true",
                      default=False, 
                      dest="sdstarchive",
                      help="Process a SDST PFRaw archive file")
    parser.add_option("--disable-gfu", 
                      action="store_false",
                      default=False, 
                      dest="GFU",
                      help="Do not run GFU filter")
    return parser

def main(options, stats={}):
    """The main L1 script"""
    tray = I3Tray()
    
    # make list of input files from GCD and infile
    '''
    if isinstance(options['infile'],list):
        infiles = [options['gcdfile']]
        infiles.extend(options['infile'])
    else:
        infiles = [options['gcdfile'], options['infile']]
    '''
    infiles=[options['gcdfile']]+glob.glob(options['infile'])
    print 'infiles: ',infiles
    
    # test access to input and output files
    for f in infiles:
        if not os.access(f,os.R_OK):
            raise Exception('Cannot read from %s'%f)
    def test_write(f):
        if f:
            try:
                open(f,'w')
            except IOError:
                raise Exception('Cannot write to %s'%f)
            finally:
                os.remove(f)
    test_write(options['outfile'])

    ########################################

    tray = I3Tray()
    
    tray.AddModule("I3Reader", "reader", filenamelist=infiles)
        
    # Ramesh. Ignore frame where only MCPrimary exist after detector simulation.
    def inframe(frame):
        #print frame.keys()
        #return 'IceTopRawData' in frame
        return 'I3TriggerHierarchy' in frame

    if options['sdstarchive']:
        # do some renaming and copying to clean up
        def trigger_cleanup(frame):
            if 'DSTTriggers' in frame and 'I3TriggerHierarchy' not in frame:
                frame['I3TriggerHierarchy'] = dataclasses.I3TriggerHierarchy.from_frame(frame,'DSTTriggers')
        tray.AddModule(trigger_cleanup,'trigger_cleanup',Streams=[icetray.I3Frame.DAQ])
        tray.Add(inframe, 'inframe',Streams=[icetray.I3Frame.DAQ]) #Ramesh. Ignore frame where only MCPrimary exist.
        
        from icecube import WaveCalibrator
        icetray.load("DomTools", False)
        icetray.load("wavedeform", False)
        icetray.load("tpx", False)
        
        # rerun IceTop calibration
        tray.AddModule("I3WaveCalibrator", name+"_WaveCalibrator_IceTop",
               Launches="IceTopRawData",
               Waveforms="IceTopCalibratedWaveforms", Errata="IceTopErrata",
               WaveformRange="")
        tray.AddModule('I3WaveformSplitter', name+'_WaveformSplitter_IceTop',
               Force=True, Input='IceTopCalibratedWaveforms',
               PickUnsaturatedATWD=True, HLC_ATWD='CalibratedIceTopATWD_HLC',
               HLC_FADC='CalibratedIceTopFADC_HLC', SLC='CalibratedIceTopATWD_SLC')
        tray.AddModule('I3TopHLCPulseExtractor', 'tpx_hlc',
               Waveforms='CalibratedIceTopATWD_HLC', #PEPulses=filter_globals.ITRecoPulses, #Ramesh
               PulseInfo='IceTopHLCPulseInfo', VEMPulses='IceTopHLCVEMPulses',
               MinimumLeadingEdgeTime = -100.0*I3Units.ns)
        tray.AddModule('I3TopSLCPulseExtractor', 'tpx_slc',
               Waveforms='CalibratedIceTopATWD_SLC', PEPulses='IceTopPulses_SLC',
               VEMPulses='IceTopSLCVEMPulses')
    
    # run online filters
    online_kwargs = {}
    if options['photonicsdir']:
        online_kwargs.update({
            'PhotonicsTabledirMu': os.path.join(options['photonicsdir'],'SPICE1'), 
            'PhotonicsDriverfileMu': os.path.join('driverfiles','mu_photorec.list'),
            'SplineRecoAmplitudeTable': os.path.join(options['photonicsdir'],'splines','InfBareMu_mie_abs_z20a10.fits'),
            'SplineRecoTimingTable': os.path.join(options['photonicsdir'],'splines','InfBareMu_mie_prob_z20a10.fits'),
            'hese_followup_base_GCD_filename': options['gcdfile'],
        })
    if options['sdstarchive']:
        online_kwargs['sdstarchive'] = True
    if options['GFU'] is not None:
        online_kwargs['gfu_enabled'] = options['GFU']

    online_kwargs['ehealert_followup'] = False #Ramesh
    online_kwargs['hese_followup'] = False #Ramesh

    # All Filter check is done here.
    tray.AddSegment(OnlineFilter, "OnlineFilter", 
                    decode=False, simulation=True,
                    vemcal_enabled=False,
                    **online_kwargs
                    )

    # make random service
    seed = os.getpid()
    filter_mask_randoms = phys_services.I3GSLRandomService(seed)

    # override MinBias Prescale
    filterconfigs = filter_globals.filter_pairs + filter_globals.sdst_pairs
    if options['MinBiasPrescale']:
        for i,filtertuple in enumerate(filterconfigs):
            if filtertuple[0] == filter_globals.FilterMinBias:
                del filterconfigs[i]
                filterconfigs.append((filtertuple[0],options['MinBiasPrescale']))
                break
    print filterconfigs
    
    # Generate filter Masks for all P frames
    tray.AddModule(filter_tools.FilterMaskMaker, "MakeFilterMasks",
                   OutputMaskName = filter_globals.filter_mask,
                   FilterConfigs = filterconfigs,
                   RandomService = filter_mask_randoms)

    # Merge the FilterMasks
    tray.AddModule("OrPframeFilterMasks", "make_q_filtermask",
                   InputName = filter_globals.filter_mask,
                   OutputName = filter_globals.qfilter_mask)


    #Q+P frame specific keep module needs to go first, as KeepFromSubstram
    #will rename things, let's rename post keep.  
    def is_Q(frame):
        return frame.Stop==frame.DAQ

    keep_before_merge = filter_globals.q_frame_keeps + [
                            'InIceDSTPulses', # keep DST pulse masks
                            'IceTopDSTPulses',
                            'CalibratedWaveformRange', # keep calibration info
                            'UncleanedInIcePulsesTimeRange',
                            'SplitUncleanedInIcePulses',
                            'SplitUncleanedInIcePulsesTimeRange',
                            'SplitUncleanedInIceDSTPulsesTimeRange',
                            'CalibrationErrata',
                            'SaturationWindows',
                            'InIceRawData', # keep raw data for now
                            'IceTopRawData',
                            'CorsikaWeightMap', # sim keys
                            'I3MCWeightDict',
                            'MCHitSeriesMap',
                            'MMCTrackList',
                            'I3MCTree',
                            'I3LinearizedMCTree',
                            'MCPrimary',
                            'MCPrimaryInfo',
                            'TimeShift', # the time shift amount
                            'WIMP_params', # Wimp-sim
                            'SimTrimmer', # for SimTrimmer flag
                            'I3MCPESeriesMap',
                            'I3MCPulseSeriesMap',
                            'I3MCPulseSeriesMapParticleIDMap',
                           ]
    if options['sdstarchive']:
        keep_before_merge.extend(['I3SuperDSTTimeRange'])
    tray.AddModule("Keep", "keep_before_merge",
                   keys = keep_before_merge,
                   If=is_Q
                   )

    ## second set of prekeeps, conditional on filter content, based on newly created Qfiltermask
    ## Determine if we should apply harsh keep for events that failed to pass any filter
    ## Note: excluding the sdst_streams entries

    tray.AddModule("I3IcePickModule<FilterMaskFilter>","filterMaskCheckAll",
                   FilterNameList = filter_globals.filter_streams,
                   FilterResultName = filter_globals.qfilter_mask,
                   DecisionName = "PassedAnyFilter",
                   DiscardEvents = False, 
                   Streams = [icetray.I3Frame.DAQ]
                   )
    def do_save_just_superdst(frame):
        if frame.Has("PassedAnyFilter"):
            if not frame["PassedAnyFilter"].value:
                return True    #  <- Event failed to pass any filter.  
            else:
                return False # <- Event passed some filter

        else:
            print "Failed to find key frame Bool!!"
            return False

    keep_only_superdsts = filter_globals.keep_nofilterpass+[
                             'PassedAnyFilter',
                             'InIceDSTPulses',
                             'IceTopDSTPulses',
                             'SplitUncleanedInIcePulses',
                             'SplitUncleanedInIcePulsesTimeRange',
                             'SplitUncleanedInIceDSTPulsesTimeRange',
                             'CorsikaWeightMap', # sim keys
                             'I3MCWeightDict',
                             'MCHitSeriesMap',
                             'MMCTrackList',
                             'I3MCTree',
                             'I3LinearizedMCTree',
                             'MCPrimary',
                             'MCPrimaryInfo',
                             'TimeShift', # the time shift amount
                             'WIMP_params', # Wimp-sim
                             'I3MCPESeriesMap',
                             'I3MCPulseSeriesMap',
                             'I3MCPulseSeriesMapParticleIDMap',
                             ]
    if options['sdstarchive']:
        keep_only_superdsts.extend(['I3SuperDSTTimeRange',
                                    ])
    tray.AddModule("Keep", "KeepOnlySuperDSTs",
                   keys = keep_only_superdsts,
                   If = do_save_just_superdst
                   )
        
    ## Now clean up the events that not even the SuperDST filters passed on.
    tray.AddModule("I3IcePickModule<FilterMaskFilter>","filterMaskCheckSDST",
                   FilterNameList = filter_globals.sdst_streams,
                   FilterResultName = filter_globals.qfilter_mask,
                   DecisionName = "PassedKeepSuperDSTOnly",
                   DiscardEvents = False,
                   Streams = [icetray.I3Frame.DAQ]
                   )

    if not options['sdstarchive']:
        def dont_save_superdst(frame):
            if frame.Has("PassedKeepSuperDSTOnly") and frame.Has("PassedAnyFilter"):
                if frame["PassedAnyFilter"].value:
                    return False  #  <- these passed a regular filter, keeper
                elif not frame["PassedKeepSuperDSTOnly"].value:
                    return True    #  <- Event failed to pass SDST filter.  
                else:
                    return False # <- Event passed some  SDST filter

            else:
                print "Failed to find key frame Bool!!"
                return False

        tray.AddModule("Keep", "KeepOnlyDSTs",
                       keys = filter_globals.keep_dst_only
                              + ["PassedAnyFilter","PassedKeepSuperDSTOnly",
                                 filter_globals.eventheader],
                              If = dont_save_superdst
                       )

    ## Frames should now contain only what is needed.  now flatten, write/send to server
    ## Squish P frames back to single Q frame, one for each split:
    tray.AddModule("KeepFromSubstream","null_stream",
                   StreamName = filter_globals.NullSplitter,
                   KeepKeys = filter_globals.null_split_keeps,
                   )

    in_ice_keeps = filter_globals.inice_split_keeps + filter_globals.onlinel2filter_keeps
    in_ice_keeps = in_ice_keeps + ['I3EventHeader',
                                   'SplitUncleanedInIcePulses',
                                   'TriggerSplitterLaunchWindow',
                                   'I3TriggerHierarchy',
                                   'GCFilter_GCFilterMJD']
    tray.AddModule("Keep", "inice_keeps",
                   keys = in_ice_keeps,
                   If = which_split(split_name=filter_globals.InIceSplitter),
                   )


    tray.AddModule("KeepFromSubstream","icetop_split_stream",
                   StreamName = filter_globals.IceTopSplitter,
                   KeepKeys = filter_globals.icetop_split_keeps,
                   )

    # Apply small keep list (SuperDST/SmallTrig/DST/FilterMask for non-filter passers
    # Remove I3DAQData object for events not passing one of the 'filters_keeping_allraw'
    tray.AddModule("I3IcePickModule<FilterMaskFilter>","filterMaskCheck",
                   FilterNameList = filter_globals.filters_keeping_allraw,
                   FilterResultName = filter_globals.qfilter_mask,
                   DecisionName = "PassedConventional",
                   DiscardEvents = False,
                   Streams = [icetray.I3Frame.DAQ]
                   )

    ## Clean out the Raw Data when not passing conventional filter
    def I3RawDataCleaner(frame):
        if not (('PassedConventional' in frame and 
                 frame['PassedConventional'].value == True) or 
                ('SimTrimmer' in frame and
                 frame['SimTrimmer'].value == True)
               ):
            frame.Delete('InIceRawData')
            frame.Delete('IceTopRawData')

    tray.AddModule(I3RawDataCleaner,"CleanErrataForConventional",
                   Streams=[icetray.I3Frame.DAQ])

    ###################################################################
    ########### WRITE STUFF                  ##########################
    ###################################################################
    
    # clean up special case for bz2 files to get around empty file bug
    bzip2_files = []
    if options['outfile'] and options['outfile'].endswith('.bz2'):
        options['outfile'] = options['outfile'][:-4]
        bzip2_files.append(options['outfile'])
    
    # Write the physics and DAQ frames
    tray.AddModule("I3Writer", "EventWriter",
                   filename=options['outfile'],
                   Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics]
                   )

    tray.AddModule("TrashCan","byebye")

    # make it go
    if options['num'] >= 0:
        tray.Execute(options['num'])
    else:
        tray.Execute()

    tray.Finish()

    # print more CPU usage info. than speicifed by default
    tray.PrintUsage(fraction=1.0) 
    for entry in tray.Usage():
        stats[entry.key()] = entry.data().usertime

    if bzip2_files:
        # now do bzip2
        subprocess.check_call(['/usr/bin/bzip2', '-f']+bzip2_files)   
 
    # clean up forcefully in case we're running this in a loop
    del tray


### iceprod stuff ###
try:
    from iceprod.modules import ipmodule
except ImportError, e:
    print 'Module iceprod.modules not found. Will not define IceProd Class'
else:
    Level1SimulationFilter = ipmodule.FromOptionParser(make_parser(),main)
### end iceprod stuff ###


if __name__ == '__main__':
    # run as script from the command line
    # get parsed args
    parser = make_parser()
    (options,args) = parser.parse_args()
    opts = {}
    # convert to dictionary
    for name in parser.defaults:
        value = getattr(options,name)
        if name == 'infile' and ',' in value:
            value = value.split(',') # split into multiple inputs
        opts[name] = value

    # call main function
    main(opts)

