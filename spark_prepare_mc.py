#!/usr/bin/env python

"""
Idea here is to split all MC simulation events randomly into two files. 
Use one of them for training and the other one for testing and predicting.
Then merge those files and create a final file.

For Example:
  Inout: 
    analysis_simulation_HLCCoreSeed_slcQcut_all.h5 ($ python booking_data_mc.py --isMC)
  Output:
    1.analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5
    2.analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5

To Run:
    $ python spark_prepare_mc.py --split
    $ python spark_prepare_mc.py --split --isqgsjet
    $ python spark_prepare_mc.py --merge
    $ python spark_prepare_mc.py --merge --isqgsjet
"""
import tables
import numpy
from optparse import OptionParser

parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)

parser.add_option("--isqgsjet", action='store_true', help="QGSJETII-04 interaction model if called.")
parser.add_option("--split", action='store_true', help="Split one simulated data file into tow equal parts.")
parser.add_option("--merge", action='store_true', help="Merge two equally splitted simulated data into one file.")

opts, args = parser.parse_args()

split    = opts.split
merge    = opts.merge
isqgsjet = opts.isqgsjet

thisdir = "/data/icet0/rkoirala/LowEnergy/RandomForest/"

if split:
    if isqgsjet:
        hfreadfile   = thisdir+'analysis_simulation_HLCCoreSeed_qgsjet_all.h5'
    else:
        hfreadfile   = thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_all.h5'
    hfstoreblockfile = thisdir+'h5_block_storage.h5'

    hf = tables.open_file(hfreadfile)
    #Keys =  ['Qmax'   , 'Qhillas'  , 'Qtotal' , 'Qtotalhlc',       
    #         'Nslc'   , 'Ntanks'   , 'Nsta'  , 
    #         'Tcorex' , 'Tcorey'   , 'COGX', 'COGY', 'T0', 
    #         'Tzenith', 'Tazimuth' , 'Plane_Zenith', 'Plane_Azimuth',
    #         'Energy' , 
    #         'LoudSta', 'FiltCondt',     
    #         'TankX'  , 'TankY'    , 'Pulses'   , 'HitsTime' 
    #         'WeightPoly',  'WeightH4a', 'WeightGST' , 
    Pulses = numpy.array(hf.root.Pulses[:])
    if isqgsjet:
        planezenith = hf.root.PlaneZenith[:]
        planeazimuth= hf.root.PlaneAzimuth[:]
        print len(hf.root.ZSC_avg[:])
    else:
        planezenith = hf.root.Plane_Zenith[:]
        planeazimuth= hf.root.Plane_Azimuth[:]

    data  = numpy.column_stack((hf.root.FiltCondt[:],
                                hf.root.LoudSta[:],
                                hf.root.Qhillas[:],
                                hf.root.Qtotal[:],
                                hf.root.Qmax[:],
                                hf.root.Qtotalhlc[:],
                                hf.root.Nsta[:],
                                hf.root.Nslc[:],
                                hf.root.Ntanks[:],
                                hf.root.Tcorex[:],
                                hf.root.Tcorey[:],
                                hf.root.COGX[:],
                                hf.root.COGY[:],
                                hf.root.Tzenith[:],
                                hf.root.Tazimuth[:],
                                Pulses,
                                numpy.array(hf.root.TankX[:]),
                                numpy.array(hf.root.TankY[:]),
                                numpy.array(hf.root.TankZ[:]),
                                numpy.array(hf.root.HitsTime[:]),
                                planezenith,
                                planeazimuth,
                                hf.root.Energy[:],
                                hf.root.T0[:],
                                hf.root.WeightPoly[:],
                                hf.root.WeightH4a[:],
                                hf.root.WeightGST[:],
                                hf.root.pdg_encoding[:],
                                numpy.sum(Pulses[:, :2], axis=1),
                                hf.root.ZSC_avg[:]
                            ))
    hf.close()

    numpy.random.shuffle(data)

    #print data[1], len(data[1]), len(Pulses[0]), data.shape, len(data)

    FiltCondt = data[:,0]
    LoudSta   = data[:,1]
    Qhillas   = data[:,2]
    Qtotal    = data[:,3]
    Qmax      = data[:,4]
    Qtotalhlc = data[:,5]
    Nsta      = data[:,6]
    Nslc      = data[:,7]
    Ntanks    = data[:,8]
    Tcorex    = data[:,9]
    Tcorey    = data[:,10]
    COGX      = data[:,11]
    COGY      = data[:,12]
    Tzenith   = data[:,13]
    Tazimuth  = data[:,14]
    Pulses    = data[:,15:15+35]
    TankX     = data[:,50:50+35]
    TankY     = data[:,85:85+35]
    TankZ     = data[:,120:120+35]
    HitsTime  = data[:,155:155+35]
    PlaneZenith  = data[:,190]
    PlaneAzimuth = data[:,191]
    Energy     = data[:,192]
    T0         = data[:,193]
    WeightPoly = data[:,194]
    WeightH4a  = data[:,195]
    WeightGST  = data[:,196]
    pdg        = data[:,197]
    Qsum2      = data[:,198]
    ZSC_avg    = data[:,199]

    # Split data into half and save them in different hdf files.
    for i in range(2):
        if i==0:
            #name = '1st'
            name = 'qgsjet_1st'
            n1 = 0
            n2 = int(len(data)/2)
        if i==1:
            #name = '2nd'
            name = 'qgsjet_2nd'
            n1 = n2
            n2 = len(data)
    
        outhdf = thisdir+"analysis_simulation_HLCCoreSeed_slcQcut_%shalf.h5"%name
        print outhdf
        hf = tables.open_file(outhdf, 'w')
        hf.create_array('/', 'Qmax', Qmax[n1:n2])
        hf.create_array('/', 'Qhillas', Qhillas[n1:n2])
        hf.create_array('/', 'Qtotal', Qtotal[n1:n2])
        hf.create_array('/', 'Qtotalhlc', Qtotalhlc[n1:n2])
        hf.create_array('/', 'T0', T0[n1:n2])
        hf.create_array('/', 'Nsta', Nsta[n1:n2])
        hf.create_array('/', 'Nslc', Nslc[n1:n2])
        hf.create_array('/', 'Ntanks', Ntanks[n1:n2])
        hf.create_array('/', 'Tcorex', Tcorex[n1:n2])
        hf.create_array('/', 'Tcorey', Tcorey[n1:n2])
        hf.create_array('/', 'COGX', COGX[n1:n2])
        hf.create_array('/', 'COGY', COGY[n1:n2])
        hf.create_array('/', 'Tzenith', Tzenith[n1:n2])
        hf.create_array('/', 'Tazimuth', Tazimuth[n1:n2])
        hf.create_array('/', 'PlaneZenith', PlaneZenith[n1:n2])
        hf.create_array('/', 'PlaneAzimuth', PlaneAzimuth[n1:n2])
        hf.create_array('/', 'Energy', Energy[n1:n2])
        hf.create_array('/', 'WeightPoly', WeightPoly[n1:n2])
        hf.create_array('/', 'WeightH4a', WeightH4a[n1:n2])
        hf.create_array('/', 'WeightGST', WeightGST[n1:n2])
        hf.create_array('/', 'FiltCondt', FiltCondt[n1:n2])
        hf.create_array('/', 'LoudSta', LoudSta[n1:n2])
        hf.create_array('/', 'TankX', TankX[n1:n2, :])
        hf.create_array('/', 'TankY', TankY[n1:n2, :])
        hf.create_array('/', 'TankZ', TankZ[n1:n2, :])
        hf.create_array('/', 'Pulses', Pulses[n1:n2, :])
        hf.create_array('/', 'HitsTime', HitsTime[n1:n2, :])
        hf.create_array('/', 'pdg_encoding', pdg[n1:n2])
        hf.create_array('/', 'Qsum2', Qsum2[n1:n2])
        hf.create_array('/', 'ZSC_avg', ZSC_avg[n1:n2])
        hf.close()

if merge:
    # Before merging:
    #     1. Spark: Train to create model for X, Y, Zen using analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5
    #     2. Spark: Train to create model for X, Y, Zen using analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5
    #     3. Spark: Predict X, Y, Zen for analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5 
    #               using models trained on analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5
    #     4. Spark: Predict X, Y, Zen for analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5 
    #               using models trained on analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5
    #     5. SKlearn: Predict energy in analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5 using X,Y,Zen 
    #            models created by training data in analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5
    #     6. SKlearn: Predict energy in analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5 using X,Y,Zen 
    #            models created by training data in analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5
    # Now you are ready to merge
    #hfinal   = thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_final.h5'
    #hread1   = thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5'
    #hread2   = thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5'
    hread1   = thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_qgsjet_1sthalf.h5'
    hread2   = thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_qgsjet_2ndhalf.h5'
    hfinal   = thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_qgsjet_final.h5'

    hf  = tables.open_file(hfinal, 'w')
    hf1 = tables.open_file(hread1)
    hf2 = tables.open_file(hread2)
    hf.create_array('/', 'Qhillas', numpy.append(hf1.root.Qhillas[:], hf2.root.Qhillas[:]))
    hf.create_array('/', 'Qtotal', numpy.append(hf1.root.Qtotal[:], hf2.root.Qtotal[:]))
    hf.create_array('/', 'Qtotalhlc', numpy.append(hf1.root.Qtotalhlc[:], hf2.root.Qtotalhlc[:]))
    hf.create_array('/', 'T0', numpy.append(hf1.root.T0[:], hf2.root.T0[:]))
    hf.create_array('/', 'Nsta', numpy.append(hf1.root.Nsta[:], hf2.root.Nsta[:]))
    hf.create_array('/', 'Nslc', numpy.append(hf1.root.Nslc[:], hf2.root.Nslc[:]))
    hf.create_array('/', 'Ntanks', numpy.append(hf1.root.Ntanks[:], hf2.root.Ntanks[:]))
    hf.create_array('/', 'Tcorex', numpy.append(hf1.root.Tcorex[:], hf2.root.Tcorex[:]))
    hf.create_array('/', 'Tcorey', numpy.append(hf1.root.Tcorey[:], hf2.root.Tcorey[:]))
    hf.create_array('/', 'COGX', numpy.append(hf1.root.COGX[:], hf2.root.COGX[:]))
    hf.create_array('/', 'COGY', numpy.append(hf1.root.COGY[:], hf2.root.COGY[:]))
    hf.create_array('/', 'Tzenith', numpy.append(hf1.root.Tzenith[:], hf2.root.Tzenith[:]))
    hf.create_array('/', 'Tazimuth', numpy.append(hf1.root.Tazimuth[:], hf2.root.Tazimuth[:]))
    hf.create_array('/', 'PlaneZenith', numpy.append(hf1.root.PlaneZenith[:], hf2.root.PlaneZenith[:]))
    hf.create_array('/', 'PlaneAzimuth', numpy.append(hf1.root.PlaneAzimuth[:], hf2.root.PlaneAzimuth[:]))
    hf.create_array('/', 'Energy', numpy.append(hf1.root.Energy[:], hf2.root.Energy[:]))
    hf.create_array('/', 'WeightPoly', numpy.append(hf1.root.WeightPoly[:], hf2.root.WeightPoly[:]))
    hf.create_array('/', 'WeightH4a', numpy.append(hf1.root.WeightH4a[:], hf2.root.WeightH4a[:]))
    hf.create_array('/', 'WeightGST', numpy.append(hf1.root.WeightGST[:], hf2.root.WeightGST[:]))
    hf.create_array('/', 'FiltCondt', numpy.append(hf1.root.FiltCondt[:], hf2.root.FiltCondt[:]))
    hf.create_array('/', 'LoudSta', numpy.append(hf1.root.LoudSta[:], hf2.root.LoudSta[:]))
    hf.create_array('/', 'TankX', numpy.vstack((hf1.root.TankX[:], hf2.root.TankX[:])))
    hf.create_array('/', 'TankY', numpy.vstack((hf1.root.TankY[:], hf2.root.TankY[:])))
    hf.create_array('/', 'TankZ', numpy.vstack((hf1.root.TankZ[:], hf2.root.TankZ[:])))
    hf.create_array('/', 'Pulses', numpy.vstack((hf1.root.Pulses[:], hf2.root.Pulses[:])))
    hf.create_array('/', 'HitsTime', numpy.vstack((hf1.root.HitsTime[:], hf2.root.HitsTime[:])))
    hf.create_array('/', 'pdg_encoding', numpy.append(hf1.root.pdg_encoding[:], hf2.root.pdg_encoding[:]))
    hf.create_array('/', 'ZSC_avg', numpy.append(hf1.root.ZSC_avg[:], hf2.root.ZSC_avg[:]))
    
    hf.create_array('/', 'PredictedX', numpy.append(hf1.root.PredictedX[:], hf2.root.PredictedX[:]))
    hf.create_array('/', 'PredictedY', numpy.append(hf1.root.PredictedY[:], hf2.root.PredictedY[:]))
    hf.create_array('/', 'PredictedZen', numpy.append(hf1.root.PredictedZen[:], hf2.root.PredictedZen[:]))
    #hf.create_array('/', 'PredictedLogEnergy_h4a', numpy.append(hf1.root.PredictedLogEnergy_h4a[:], hf2.root.PredictedLogEnergy_h4a[:]))
    #hf.create_array('/', 'maskTrainingEnergy', numpy.append(hf1.root.maskTrainingEnergy_h4a[:], hf2.root.maskTrainingEnergy_h4a[:]))

    hf1.close()
    hf2.close()
    hf.close()
    print 'DONE'
    
