#!/usr/bin/env python

"""
@asterix
Parallize machine learning using pyspark.
Use h5spark to convert hdf files to spark's RDD data format. 
Predict X, Y, and Zenith of experimental event.
Afte predicting x, y, and zenith using two models developed for each half of simulated data,
take average of both predicted value.
"""
def predict(what='', hdfreadfile='', hdfwritefile='', name='', varname=''):
    partition = 420

    print ''
    what_list = ['x', 'y', 'zenith', 'energy']
    hfstoreblockfile = '/data/icet0/rkoirala/LowEnergy/RandomForest/h5spark_block_storage.h5'  

    if (what=='') or (what not in what_list):
        raise Exception("Please provide what to predict. Ex: what = 'x', 'y', 'zenith', 'energy'")

    # ====================================================================================
    # Use RFModel on experimental data to get X position.        
    elif what=='x':
        print 'Predicting X'
        hfnew = tables.open_file(hfstoreblockfile, 'w') # Size: {493536, 654}
        hfnew.create_array('/', 'Block', 
                        numpy.column_stack((COGX,
                                            COGY,
                                            numpy.log10(Nsta),
                                            numpy.cos(PlaneZenith),
                                            Pulses,
                                            TankX,
                                            TankY
                                         ))
                        ) # Close block array
        hfnew.close()
        
        predicted_var_name = 'PredictedX'+varname
        rfPath = "/data/icet0/rkoirala/LowEnergy/RandomForest/RFcorexModel"+name
        
    # ====================================================================================  
    # Use RFModel on experimental data to get Y position.  
    elif what=='y':
        print 'Predicting Y'
        #hf = tables.open_file(hdfwritefile) 
        #PredictedX = hf.root.PredictedX[:]    
        #hf.close()
    
        hfnew = tables.open_file(hfstoreblockfile, 'w') # Size: {493536, 654}
        hfnew.create_array('/', 'Block', 
                        numpy.column_stack((COGX,
                                            COGY,
                                            numpy.log10(Nsta),
                                            numpy.cos(PlaneZenith),
                                            Pulses,
                                            TankX,
                                            TankY
                                         ))
                        ) # Close block array
        hfnew.close()
        
        predicted_var_name = 'PredictedY'+varname
        rfPath = "/data/icet0/rkoirala/LowEnergy/RandomForest/RFcoreyModel"+name
      
    # ====================================================================================  
    # Use RFModel on experimental data to get zenith.   
    elif what=='zenith':
        print 'Predicting Zenith'
        hf = tables.open_file(hdfwritefile) 
        ZSC_avg = hf.root.ZSC_avg[:] 
        hf.close()

        #       eprint len(PredictedX), len(PredictedY), len(numpy.log10(Ntanks)), len(PlaneZenith), len(PlaneAzimuth), len(T0), len(ZSC_avg), len(Pulses), len(HitsTime)
        hfnew = tables.open_file(hfstoreblockfile, 'w') # Size: {493536, 654}
        hfnew.create_array('/', 'Block', 
                        numpy.column_stack((COGX,
                                            COGY,
                                            numpy.log10(Nsta),
                                            PlaneZenith,
                                            PlaneAzimuth,
                                            T0,
                                            ZSC_avg, 
                                            Pulses,
                                            HitsTime
                                         ))
                        ) # Close block array
        hfnew.close()
        
        predicted_var_name = 'PredictedZen'+varname
        rfPath = "/data/icet0/rkoirala/LowEnergy/RandomForest/RFzenithModel"+name
            
    # ====================================================================================  
    # Use RFModel to predict on given variable.
    print 'Working with model', rfPath
    bestModel = PipelineModel.load(rfPath)
    print 'Model Loaded'
    print ''
    
    # ====================================================================================
    vars_dtype = {'features':VectorUDT()}
    Keys = ['features']
    rdd  = read.h5read(sc,(hfstoreblockfile,'Block'), mode='single', partitions=partition)              
    rdd  = rdd.map(lambda ar: [Vectors.dense(ar.tolist()[:])])
    # ====================================================================================
    
    schema = StructType([StructField(key, vars_dtype[key], True) for key in Keys])
    df     = spark.createDataFrame(rdd, schema)

    Prediction          = bestModel.transform(df).select("prediction").rdd
    predicted_var_value = numpy.array(Prediction.map(lambda lp: lp[0]).collect())

    hf = tables.open_file(hdfwritefile, 'a') 
    if predicted_var_name in hf.root:
        hf.remove_node('/', predicted_var_name)
    hf.create_array('/', predicted_var_name, predicted_var_value)    
    hf.close()
    # ===============================Function Done========================================
    
# Now predict x, y, zenith, and energy here.
if __name__ == "__main__":

    print 'Random Forest Regression with PySpark'
    print '-------------------------------------------------'
    print ''
    """
    To Run:
        python spark_predict_data.py -f filename.hdf --which_half 1st
    """
    from time import time
    start = time()

    # Import all packages required.
    import numpy, os, glob, tables
    from pyspark import SparkContext, SparkConf
    from pyspark.sql.session import SparkSession
    from pyspark.sql import SQLContext
    import read # from h5spark package

    from pyspark.sql.types import * # FloatType, IntegerType, ArrayType etc
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.linalg import Vectors, VectorUDT # save Pulses as vector so that it can be merged easily later for regression.

    from optparse import OptionParser
    parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
    parser.add_option('-f', '--filename', help="input experimental data filename.")
    parser.add_option('--isqgsjet', action='store_true', help="is this simulation data?")
    parser.add_option("--which_half",choices=('1st', '2nd'), dest="which_half", help="1st half or 2nd half of simulation data?")

    opts, args = parser.parse_args()

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # ========================================================================================
    # Initialise spark session
    hdfreadfile  = opts.filename
    hdfwritefile = hdfreadfile
    hf           = tables.open_file(hdfreadfile, 'a')
    Pulses   = hf.root.Pulses[:]
    TankX    = hf.root.TankX[:]
    TankY    = hf.root.TankY[:]
    HitsTime = hf.root.HitsTime[:]
    Nsta        = hf.root.Nstation.cols.value[:]
    COGX        = hf.root.ShowerCOG.cols.x[:]
    COGY        = hf.root.ShowerCOG.cols.y[:]
    PlaneZenith = hf.root.ShowerPlane.cols.zenith[:]
    PlaneAzimuth= hf.root.ShowerPlane.cols.azimuth[:]
    T0          = hf.root.ShowerCOG.cols.time[:]
    filterSTA2  = hf.root.IceTop_TwoStationFilter.cols.value[:]
    hf.close()
    
    if opts.which_half=='1st':
        if opts.isqgsjet:
            name    = '_qgsjet_1sthalfMC'
            varname = name
        else:
            name    = '_1sthalfMC'
            varname = '1st'

    elif opts.which_half=='2nd':
        if opts.isqgsjet:
            name    = '_qgsjet_2ndhalfMC'
            varname = name
        else:
            name    = '_2ndhalfMC'
            varname = '2nd'

    else:
        raise Exception('Please provide which half of model do you want to use. Options: 1st and 2nd')

    # Now predict x, y, zenith, and energy here.
    print 'Predicting x'
    predict(what='x', hdfreadfile=hdfreadfile, hdfwritefile=hdfwritefile, name=name)
    
    print 'Predicting y'
    predict(what='y', hdfreadfile=hdfreadfile, hdfwritefile=hdfwritefile, name=name)
    
    '''
    # Do this only once. The file might already has 'ZSC_avg'.
    print 'calculating zsc_avg'
    os.system('python zsc_avg.py --isExp --filename '+hdfreadfile)  
    '''
    print 'Predicting zenith'
    predict(what='zenith', hdfreadfile=hdfreadfile, hdfwritefile=hdfwritefile, name=name)
    
    print 'done with this run'
    print ''

    print ''
    total_time = time() - start    
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    print "Total running time: %dh:%dm:%ds.\n"  % (hours, mins, secs)
    print ' ---------------------------------------------------------------'
    
    spark.stop()
    # ============================DONE========================================================