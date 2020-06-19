#!/usr/bin/env python

"""
@asterix
Parallize machine learning using pyspark.
Use h5spark to convert hdf files to spark's RDD data format. 
contact: Ramesh Koirala (rkoirala@udel.edu)

thisdir = '/data/icet0/rkoirala/LowEnergy/RandomForest/'
Source of data used in this analysis is: thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_all.h5'
To generate hdf of source data, Run:
    cd /data/icet0/rkoirala/LowEnergy/RandomForest/
    python booking_data_mc.py --isMC
 
Idea here is to split all MC simulation events randomly into two files. 
Use one of them to training and the other one for testing and predicting.   
To split 'analysis_simulation_HLCCoreSeed_slcQcut_all.h5' data randomly in two files:
    python spark_prepare_mc.py
    
    Output:
        1.thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5'
        2.thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5'
        
Once the data is splitted randomly into two files, use one to train the spark random
forest regression, and use the other one to test and predit.
For training purpose thisdir+'analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5' is used.
Training X:
    TrueX
    COGX
    COGY
    logNstation
    cosTheta
    Qhit        (nevents, 35)
    TankX       (nevents, 35)      
    TankY       (nevents, 35)
    
Training Y:
    TrueY
    COGX
    COGY
    logNstation
    cosTheta
    Qhit        (nevents, 35)
    TankX       (nevents, 35)      
    TankY       (nevents, 35)

Training Zenith:
    TrueZenith
    COGX
    COGY
    logNstation
    Zenith
    Azimuth
    T0
    ZSC_avg
    Qhit        (nevents, 35)
    TankX       (nevents, 35)      
    TankY       (nevents, 35)

See Previous Version: 'spark_cv_x.py'
"""

print 'Random Forest Regression with PySpark'
print '-------------------------------------------------'
print ''

def train_x(hfreadfile='',
            trial=False  , 
            name='_jpt'
            ):
    # ========================================================================================
    # https://github.com/apache/spark/blob/master/docs/mllib-decision-tree.md
    # Define parameters for parameter tuning.
    from time import time
    start = time()
    
    print 'Training on X'
    #name      = '_2ndhalfMC' #'_leq35', '_slcQcut', _halfMC
    l         = 35
    numFolds  = 3
    partition = 420
    maxBins   = 50 # default=32

    if trial:
        numTrees            = [200]
        maxDepth            = [14]
        minInstancesPerNode = [10]
    else:
        numTrees            = [1000]
        maxDepth            = [14]
        minInstancesPerNode = [10]

    # ==========================================================
    # Note: Input hfreadfile has already been randomly shuffled.
    print 'Reading HDF File '+hfreadfile
    rfrdir           = '/data/icet0/rkoirala/LowEnergy/RandomForest/'
    #hfreadfile      = [rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5',
    #                   rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5']
    hfstoreblockfile = rfrdir+'h5spark_block_storage.h5'

    if hfreadfile != '':
        hf = tables.open_file(hfreadfile)
        FiltCondt = hf.root.FiltCondt[:]
        Qsum2     = hf.root.Qsum2[:]
        Qtotalhlc = hf.root.Qtotalhlc[:]
        Tcorex    = hf.root.Tcorex[:]
        COGX      = hf.root.COGX[:]
        COGY      = hf.root.COGY[:]
        Nsta      = hf.root.Nsta[:]
        PlaneZenith  = hf.root.PlaneZenith[:]
        Pulses    = numpy.array(hf.root.Pulses[:])
        TankX     = numpy.array(hf.root.TankX[:])
        TankY     = numpy.array(hf.root.TankY[:])
        hf.close()
    else:
        raise Exception("Please provide a proper file. You gave %s."%hfreadfile)

    # =========================================================
    # Define Mask
    mask1 = FiltCondt==1
    mask2 = (Qsum2<=0.95*Qtotalhlc)
    mask= mask1*mask2

    # =========================================================
    """
    Column stack numpy array to convert it to RDD using h5read package.
    Pulses: Qhit arranged in descending order (highest 35 Qhit).
    TankX : X-coordinate of hit tank arranged by descending order of Qhit in tanks. 
    TankY : Y-coordinate of hit tank arranged by descending order of Qhit in tanks. 
    """
    print 'num after cut: ', numpy.sum(mask)
    print Pulses.shape, mask.shape
    hfnew = tables.open_file(hfstoreblockfile, 'w') # Size: {493536, 654}
    hfnew.create_array('/', 'Block', 
                    numpy.column_stack((Tcorex[mask],
                                        COGX[mask],
                                        COGY[mask],
                                        numpy.log10(Nsta[mask]),
                                        numpy.cos(PlaneZenith[mask]),
                                        Pulses[mask],
                                        TankX[mask],
                                        TankY[mask],
                                     ))
                    ) # Close block array
    hfnew.close()

    # ========================================================================================
    # PRODUCING RDD FROM HDF FILE.
    # Convert hdf file to Sparks datatype RDD using h5spark.
    # Module "read" comes from a package called h5spark.
    # mode="multi" for many files.
    # Spark does not like numpy array but likes python list. h5read converts ndarray to spark's RDD.
    vars_dtype = {'label':FloatType(), 'features':VectorUDT()}
    Keys = ['label', 'features']
    rdd  = read.h5read(sc,(hfstoreblockfile,'Block'), mode='single', partitions=partition)            
    rdd  = rdd.map(lambda ar: [ar.tolist()[0], Vectors.dense(ar.tolist()[1:])])
    print("Number of partitions: {}".format(rdd.getNumPartitions()))

    # Creating DataFrame from RDD.
    # Schema defines the title of each column on RDD. Required to convert RDD to DataFrame.
    schema       = StructType([StructField(key, vars_dtype[key], True) for key in Keys])
    dfo          = spark.createDataFrame(rdd, schema)
    trainingData = dfo.orderBy(rand())
    #trainingData, testData = df.randomSplit([0.7, 0.3])
  
    # ========================================================================================
    # TRAIN MC DATA USING "RandomForestRegressor".
    # featureSubsetStrategy: Number of features to consider for splits at each node. 
    #   Supported: "auto", "all", "sqrt", "log2", "onethird". 
    #   If "auto": if numTrees = 1, set to "all"; 
    #              if numTrees > 1 (forest) set to "sqrt" for classification 
    #                                              "onethird" for regression. 
    # numTrees: [50, 100, 200, 300, 400]
    # best numTrees: 300
    # best maxDepth: 12
    modelType = RandomForestRegressor()
    paramGrid = (ParamGridBuilder() \
             .baseOn({modelType.labelCol   : 'label'})         \
             .baseOn({modelType.featuresCol: 'features'})      \
             .baseOn([modelType.predictionCol, 'prediction'])  \
             .baseOn([modelType.maxBins, maxBins])             \
             .addGrid(modelType.numTrees, numTrees)            \
             .addGrid(modelType.maxDepth, maxDepth)            \
             .addGrid(modelType.minInstancesPerNode, minInstancesPerNode) \
             .build())
         
    # Put all intended work in a pipleline for crossvalidator to process in sequence.         
    pipeline = Pipeline(stages=[modelType])

    # Instantiate cross validator.
    cv = CrossValidator(estimator=pipeline,
                        evaluator=RegressionEvaluator(),
                        estimatorParamMaps=paramGrid,
                        numFolds=numFolds
                        )
                                       
    # Now go ML and show your magic. All heavy lifting is done here.  
    print "Training models with trainingData."                 
    cvModel = cv.fit(trainingData)

    # Make predictions on test data. cvModel uses the best model found.
    bestModel = cvModel.bestModel

    # ========================================================================================
    # Save & load the Random Forest model
    rfPath = "/data/icet0/rkoirala/LowEnergy/RandomForest/RFcorexModel"+name
    bestModel.write().overwrite().save(rfPath)
    #samebestModel = PipelineModel.load(rfPath)

    # ========================================================================================

    featureImportances = bestModel.stages[-1].featureImportances.toArray()

    # Save featureImportances on a table.
    #from tables import *
    class Particle(IsDescription):
        cogx        = Float64Col()
        cogy        = Float64Col()
        lognstation = Float64Col()
        costheta    = Float64Col()
        pulses      = Float64Col()
        tankx       = Float64Col()
        tanky       = Float64Col()

    hf = tables.open_file(hfreadfile, 'a')
    if 'featureImportancesXAll' in hf.root:
        hf.remove_node('/', 'featureImportancesXAll')
    if 'featureImportancesX' in hf.root:
        hf.remove_node('/', 'featureImportancesX')
    hf.create_array('/', 'featureImportancesXAll', featureImportances)
    table = hf.create_table('/', 'featureImportancesX', Particle, "feature importances for X")
    particle = table.row
    particle['cogx']      =  featureImportances[0]
    particle['cogy']      =  featureImportances[1]
    particle['lognstation'] =  featureImportances[2]
    particle['costheta']  =  featureImportances[3]
    particle['pulses']    =   numpy.sum(featureImportances[int(4+0*l):int(4+1*l)])
    particle['tankx']     =   numpy.sum(featureImportances[int(4+1*l):int(4+2*l)])
    particle['tanky']     =   numpy.sum(featureImportances[int(4+2*l):int(4+3*l)])
    particle.append()
    table.flush()
    hf.close()

    # ========================================================================================
    # Miscellaneous activities
    print ''
    print '---------------------------------------------------------------'
    print "Feature Importance: ", ' cogx      :' , 100.*featureImportances[0]
    print "                    ", ' cogy      :' , 100.*featureImportances[1]
    print "                    ", ' logNstation:', 100.*featureImportances[2]
    print "                    ", ' cosTheta :' , 100.*featureImportances[3]
    print "                    ", ' Pulses   :' , 100.*numpy.sum(featureImportances[int(4+0*l):int(4+1*l)])
    print "                    ", ' TankX    :' , 100.*numpy.sum(featureImportances[int(4+1*l):int(4+2*l)])
    print "                    ", ' TankY    :' , 100.*numpy.sum(featureImportances[int(4+2*l):int(4+3*l)])

    print ''
    print '---------------------------------------------------------------'

    print ''
    print '------------------------------------------------------------'
    total_time = time() - start    
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    print "X running time: %dh:%dm:%ds.\n"  % (hours, mins, secs)
    print '------------------------------------------------------------'

    print '---------------------------------------------------------------'

    print ''
    print 'numFolds: ', numFolds
    print 'numTrees: ', numTrees
    print 'maxDepth: ', maxDepth
    print 'maxBins : ', maxBins
    print 'minInstancesPerNode: ', minInstancesPerNode
    print 'partition: ', partition
    print 'cv model   : ', cvModel
    print 'best model : ', cvModel.bestModel
    print 'avg metric : ', cvModel.avgMetrics
    print '------------------------------------------------------------'

def train_y(hfreadfile='',
            trial=False  , 
            name='_jpt'
            ):
    # ========================================================================================
    # https://github.com/apache/spark/blob/master/docs/mllib-decision-tree.md
    # Define parameters for parameter tuning.
    from time import time
    start = time()
    
    print 'Training on Y'
    #name      = '_2ndhalfMC' #'_leq35', '_slcQcut', _halfMC
    l         = 35
    numFolds  = 3
    partition = 420
    maxBins   = 50 # default=32

    if trial:
        numTrees            = [200]
        maxDepth            = [14]
        minInstancesPerNode = [10]
    else:
        numTrees            = [1000]
        maxDepth            = [14]
        minInstancesPerNode = [10]

    # ==========================================================
    # Note: Input hfreadfile has already been randomly shuffled.
    print 'Reading HDF File '+hfreadfile
    rfrdir            = '/data/icet0/rkoirala/LowEnergy/RandomForest/'
    #hfreadfile       = [rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5',
    #                    rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5']
    hfstoreblockfile = rfrdir+'h5spark_block_storage.h5'

    if hfreadfile != '':
        hf = tables.open_file(hfreadfile)
        FiltCondt = hf.root.FiltCondt[:]
        Qsum2     = hf.root.Qsum2[:]
        Qtotalhlc = hf.root.Qtotalhlc[:]
        Tcorey    = hf.root.Tcorey[:]
        COGX      = hf.root.COGX[:]
        COGY      = hf.root.COGY[:]
        Nsta      = hf.root.Nsta[:]
        PlaneZenith  = hf.root.PlaneZenith[:]
        Pulses    = numpy.array(hf.root.Pulses[:])
        TankX     = numpy.array(hf.root.TankX[:])
        TankY     = numpy.array(hf.root.TankY[:])
        hf.close()
    else:
        raise Exception("Please provide a proper file. You gave %s."%hfreadfile)
    
    # ========================================================================================
    # Define Mask
    mask1 = FiltCondt==1
    mask2 = (Qsum2<=0.95*Qtotalhlc)
    mask= mask1*mask2

    # ========================================================================================
    """
    Column stack numpy array to convert it to RDD using h5read package.
    Pulses: Qhit arranged in descending order (highest 35 Qhit).
    TankX : X-coordinate of hit tank arranged by descending order of Qhit in tanks. 
    TankY : Y-coordinate of hit tank arranged by descending order of Qhit in tanks. 
    """
    print 'num after cut: ', numpy.sum(mask)
    print Pulses.shape, mask.shape
    hfnew = tables.open_file(hfstoreblockfile, 'w') # Size: {493536, 654}
    hfnew.create_array('/', 'Block', 
                    numpy.column_stack((Tcorey[mask],
                                        COGX[mask],
                                        COGY[mask],
                                        numpy.log10(Nsta[mask]),
                                        numpy.cos(PlaneZenith[mask]),
                                        Pulses[mask],
                                        TankX[mask],
                                        TankY[mask],
                                     ))
                    ) # Close block array
    hfnew.close()

    # ========================================================================================
    # PRODUCING RDD FROM HDF FILE.
    # Convert hdf file to Sparks datatype RDD using h5spark.
    # Module "read" comes from a package called h5spark.
    # mode="multi" for many files.
    # Spark does not like numpy array but like python list. h5read converts ndarray to spark's RDD.
    vars_dtype = {'label':FloatType(), 'features':VectorUDT()}
    Keys = ['label', 'features']
    rdd  = read.h5read(sc,(hfstoreblockfile,'Block'), mode='single', partitions=partition)            
    rdd  = rdd.map(lambda ar: [ar.tolist()[0], Vectors.dense(ar.tolist()[1:])])
    print("Number of partitions: {}".format(rdd.getNumPartitions()))

    # Creating DataFrame from RDD.
    # Schema defines the title of each column on RDD. Required to convert RDD to DataFrame.
    schema       = StructType([StructField(key, vars_dtype[key], True) for key in Keys])
    dfo          = spark.createDataFrame(rdd, schema)
    trainingData = dfo.orderBy(rand())
    #trainingData, testData = df.randomSplit([0.7, 0.3])
  
    # ========================================================================================
    # TRAIN MC DATA USING "RandomForestRegressor".
    # featureSubsetStrategy: Number of features to consider for splits at each node. 
    #   Supported: "auto", "all", "sqrt", "log2", "onethird". 
    #   If "auto": if numTrees = 1, set to "all"; 
    #              if numTrees > 1 (forest) set to "sqrt" for classification 
    #                                              "onethird" for regression. 
    # numTrees: [50, 100, 200, 400]
    # best numTrees: 300
    # best maxDepth: 12
    modelType = RandomForestRegressor()
    paramGrid = (ParamGridBuilder() \
             .baseOn({modelType.labelCol   : 'label'})         \
             .baseOn({modelType.featuresCol: 'features'})      \
             .baseOn([modelType.predictionCol, 'prediction'])  \
             .baseOn([modelType.maxBins, maxBins])             \
             .addGrid(modelType.numTrees, numTrees)            \
             .addGrid(modelType.maxDepth, maxDepth)            \
             .addGrid(modelType.minInstancesPerNode, minInstancesPerNode) \
             .build())
         
    # Put all intended work in a pipleline for crossvalidator to process in sequence.         
    pipeline = Pipeline(stages=[modelType])

    # Instantiate cross validator.
    cv = CrossValidator(estimator=pipeline,
                        evaluator=RegressionEvaluator(),
                        estimatorParamMaps=paramGrid,
                        numFolds=numFolds
                        )
                                       
    # Now go ML and show your magic. All heavy lifting is done here.  
    print "Training models with trainingData."                 
    cvModel = cv.fit(trainingData)
    bestModel = cvModel.bestModel
    # ========================================================================================
    # If you want to predict using test data.
    #trainPredictionsAndLabels = bestModel.transform(testData).select("label", "prediction").rdd
    #PredictedZen = numpy.array(trainPredictionsAndLabels.map(lambda lp: lp[1]).collect())

    # ========================================================================================
    # Save & load the Random Forest model
    rfPath = "/data/icet0/rkoirala/LowEnergy/RandomForest/RFcoreyModel"+name
    bestModel.write().overwrite().save(rfPath)
    #samebestModel = PipelineModel.load(rfPath)

    # ========================================================================================
    featureImportances = bestModel.stages[-1].featureImportances.toArray()

    # Save featureImportances on a table.
    #from tables import *
    class Particle(IsDescription):
        cogx        = Float64Col()
        cogy        = Float64Col()
        lognstation = Float64Col()
        costheta    = Float64Col()
        pulses      = Float64Col()
        tankx       = Float64Col()
        tanky       = Float64Col()

    hf = tables.open_file(hfreadfile, 'a')
    if 'featureImportancesYAll' in hf.root:
        hf.remove_node('/', 'featureImportancesYAll')
    if 'featureImportancesY' in hf.root:
        hf.remove_node('/', 'featureImportancesY')
    hf.create_array('/', 'featureImportancesYAll', featureImportances)
    table = hf.create_table('/', 'featureImportancesY', Particle, "feature importances for Y")
    particle = table.row
    particle['cogx']      =  featureImportances[0]
    particle['cogy']      =  featureImportances[1]
    particle['lognstation'] =  featureImportances[2]
    particle['costheta']  =  featureImportances[3]
    particle['pulses']    =   numpy.sum(featureImportances[int(4+0*l):int(4+1*l)])
    particle['tankx']     =   numpy.sum(featureImportances[int(4+1*l):int(4+2*l)])
    particle['tanky']     =   numpy.sum(featureImportances[int(4+2*l):int(4+3*l)])
    particle.append()
    table.flush()
    hf.close()

    # ========================================================================================
    # Miscellaneous activities
    print ''
    print '---------------------------------------------------------------'
    print "Feature Importance: ", ' cogx      :' , 100.*featureImportances[0]
    print "                    ", ' cogy      :' , 100.*featureImportances[1]
    print "                    ", ' logNstation:', 100.*featureImportances[2]
    print "                    ", ' cosTheta :' , 100.*featureImportances[3]
    print "                    ", ' Pulses   :' , 100.*numpy.sum(featureImportances[int(4+0*l):int(4+1*l)])
    print "                    ", ' TankX    :' , 100.*numpy.sum(featureImportances[int(4+1*l):int(4+2*l)])
    print "                    ", ' TankY    :' , 100.*numpy.sum(featureImportances[int(4+2*l):int(4+3*l)])

    print ''
    print '---------------------------------------------------------------'

    print ''
    print '------------------------------------------------------------'
    total_time = time() - start    
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    print "Y running time: %dh:%dm:%ds.\n"  % (hours, mins, secs)
    print '------------------------------------------------------------'

    print '---------------------------------------------------------------'

    print ''
    print 'numFolds: ', numFolds
    print 'numTrees: ', numTrees
    print 'maxDepth: ', maxDepth
    print 'maxBins : ', maxBins
    print 'minInstancesPerNode: ', minInstancesPerNode
    print 'partition: ', partition
    print 'cv model   : ', cvModel
    print 'best model : ', cvModel.bestModel
    print 'avg metric : ', cvModel.avgMetrics
    print '------------------------------------------------------------'

def train_theta(hfreadfile='',
                trial=False  , 
                name='_jpt'
                ):
                
    from time import time
    start = time()

    print 'Training on theta'
    l         = 35
    numFolds  = 3
    partition = 420
    maxBins   = 50 # default=32

    if trial:
        numTrees            = [200]
        maxDepth            = [14]
        minInstancesPerNode = [10]
    else:
        numTrees            = [1000]
        maxDepth            = [14]
        minInstancesPerNode = [10]

    # ==========================================================
    # Note: Input hfreadfile has already been randomly shuffled.
    print 'Reading HDF File '+hfreadfile
    rfrdir            = '/data/icet0/rkoirala/LowEnergy/RandomForest/'
    #hfreadfile       = [rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5',
    #                    rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5']
    hfstoreblockfile = rfrdir+'h5spark_block_storage.h5'

    if hfreadfile != '':
        hf = tables.open_file(hfreadfile)
        FiltCondt   = hf.root.FiltCondt[:]
        Qsum2       = hf.root.Qsum2[:]
        Qtotalhlc   = hf.root.Qtotalhlc[:]
        Tzenith     = hf.root.Tzenith[:]
        COGX        = hf.root.COGX[:]
        COGY        = hf.root.COGY[:]
        Nsta        = hf.root.Nsta[:]
        PlaneZenith = hf.root.PlaneZenith[:]
        PlaneAzimuth= hf.root.PlaneAzimuth[:]
        T0          = hf.root.T0[:]
        ZSC_avg     = hf.root.ZSC_avg[:]
        Pulses      = hf.root.Pulses[:]
        HitsTime    = hf.root.HitsTime[:]
        hf.close()
    else:
        raise Exception("Please provide a proper file. You gave %s."%hfreadfile)

    # ========================================================================================
    # Define Mask
    mask1 = FiltCondt==1
    mask2 = (Qsum2<=0.95*Qtotalhlc)
    mask= mask1*mask2

    # ========================================================================================
    """ 
    Column stack numpy array to convert it to RDD using h5read package.
    Pulses: Qhit arranged in descending order (highest 35 Qhit).
    HitsTime: Relative hit time arranged with first hit first (first 35 hits).
    """
    print 'num after cut: ', numpy.sum(mask)

    hfnew = tables.open_file(hfstoreblockfile, 'w') # Size: {493536, 654}
    hfnew.create_array('/', 'Block', 
                    numpy.column_stack((Tzenith[mask],
                                        COGX[mask],
                                        COGY[mask],
                                        numpy.log10(Nsta[mask]),
                                        PlaneZenith[mask],
                                        PlaneAzimuth[mask],
                                        T0[mask],
                                        ZSC_avg[mask], 
                                        Pulses[mask],
                                        HitsTime[mask],
                                     ))
                    ) # Close block array
    hfnew.close()

    # ========================================================================================
    # PRODUCING RDD FROM HDF FILE.
    # Convert hdf file to Sparks datatype RDD using h5spark.
    # Module "read" comes from a package called h5spark.
    # mode="multi" for many files.
    # Spark does not like numpy array but like python list. h5read converts ndarray to spark's RDD.
    vars_dtype = {'label':FloatType(), 'features':VectorUDT()}
    Keys = ['label', 'features']
    rdd=read.h5read(sc,(hfstoreblockfile,'Block'), 
                    mode='single', partitions=partition)
                
    rdd   = rdd.map(lambda ar: [ar.tolist()[0], Vectors.dense(ar.tolist()[1:])])        
    print("Number of partitions: {}".format(rdd.getNumPartitions()))

    # Schema defines the title of each column on RDD. Required to convert RDD to DataFrame.
    # Convert RDD to DataFrame for CrossValidator to use.
    schema       = StructType([StructField(key, vars_dtype[key], True) for key in Keys])
    dfo          = spark.createDataFrame(rdd, schema)
    trainingData = dfo.orderBy(rand())
    #trainingData, testData = df.randomSplit([0.7, 0.3])


    # ========================================================================================
    # TRAIN MC DATA USING "RandomForestRegressor".
    # featureSubsetStrategy: Number of features to consider for splits at each node. 
    #   Supported: "auto", "all", "sqrt", "log2", "onethird". 
    #   If "auto": if numTrees = 1, set to "all"; 
    #              if numTrees > 1 (forest) set to "sqrt" for classification 
    #                                              "onethird" for regression. 
    modelType = RandomForestRegressor()
    paramGrid = (ParamGridBuilder() \
             .baseOn({modelType.labelCol   : 'label'})         \
             .baseOn({modelType.featuresCol: 'features'})      \
             .baseOn([modelType.predictionCol, 'prediction'])  \
             .baseOn([modelType.maxBins, maxBins])             \
             .addGrid(modelType.numTrees, numTrees)            \
             .addGrid(modelType.maxDepth, maxDepth)            \
             .addGrid(modelType.minInstancesPerNode, minInstancesPerNode) \
             .build())
         
    # Put all intended work in a pipleline for crossvalidator to process in sequence.         
    pipeline = Pipeline(stages=[modelType])

    # Instantiate cross validator.
    cv = CrossValidator(estimator=pipeline,
                        evaluator=RegressionEvaluator(),
                        estimatorParamMaps=paramGrid,
                        #parallelism=4,
                        numFolds=numFolds
                        )
                                       
    # Now go ML and show your magic. All heavy lifting is done here.  
    print "Training models with trainingData."                 
    cvModel = cv.fit(trainingData)
    bestModel = cvModel.bestModel # this is no need because cvModel=bestModel by default.
    # ========================================================================================
    # If you want to predict using test data.
    #trainPredictionsAndLabels = bestModel.transform(testData).select("label", "prediction").rdd
    #PredictedZen = numpy.array(trainPredictionsAndLabels.map(lambda lp: lp[1]).collect())

    # ========================================================================================
    # Save & load the Random Forest model
    rfPath = "/data/icet0/rkoirala/LowEnergy/RandomForest/RFzenithModel"+name
    bestModel.write().overwrite().save(rfPath)
    #samebestModel = PipelineModel.load(rfPath)

    # ========================================================================================
    featureImportances = bestModel.stages[-1].featureImportances.toArray()

    # ===============================================================
    # Save featureImportances on a table.
    #from tables import IsDescription, Float64Col
    class Particle(IsDescription):
        cogx        = Float64Col()
        cogy        = Float64Col()
        lognstation = Float64Col()
        planetheta  = Float64Col()
        planeazimuth= Float64Col()
        t0          = Float64Col()
        zscavg      = Float64Col()
        pulses      = Float64Col()
        hitstime    = Float64Col()

    hf = tables.open_file(hfreadfile, 'a')
    if 'featureImportancesZen' in hf.root:
        hf.remove_node('/', 'featureImportancesZen')
    if 'featureImportancesZenAll' in hf.root:
        hf.remove_node('/', 'featureImportancesZenAll')
    hf.create_array('/', 'featureImportancesZenAll', featureImportances)
    table = hf.create_table('/', 'featureImportancesZen', Particle, "feature importances for Zenith")
    particle = table.row
    particle['cogx']         =  featureImportances[0]
    particle['cogy']         =  featureImportances[1]
    particle['lognstation']  =  featureImportances[2]
    particle['planetheta']   =  featureImportances[3]
    particle['planeazimuth'] =  featureImportances[4]
    particle['t0']           =  featureImportances[5]
    particle['zscavg']       =  featureImportances[6]
    particle['pulses']       =  numpy.sum(featureImportances[int(7+0*l):int(7+1*l)])
    particle['hitstime']     =  numpy.sum(featureImportances[int(7+1*l):int(7+2*l)])
    particle.append()
    table.flush()
    hf.close()

    # ========================================================================================
    # Miscellaneous activities
    print ''
    print '---------------------------------------------------------------'
    print "Feature Importance: ", 'cogx: ' , 100*featureImportances[0]
    print "                    ", 'cogy: ' , 100*featureImportances[1]
    print "                    ", 'logNsta   : ' , 100*featureImportances[2]
    print "                    ", 'PlaneTheta: ' , 100*featureImportances[3]
    print "                    ", 'PlaneAzimu: ' , 100*featureImportances[4]
    print "                    ", 't0        : ' , 100*featureImportances[5]
    print "                    ", 'ZSC_avg   : ' , 100*featureImportances[6]
    print "                    ", 'Pulses    : ' , 100*numpy.sum(featureImportances[int(7+0*l):int(7+1*l)])
    print "                    ", 'HitsTime  : ' , 100*numpy.sum(featureImportances[int(7+1*l):int(7+2*l)])

    print ''
    print ' ---------------------------------------------------------------'
    total_time = time() - start    
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    print "Zenith running time: %dh:%dm:%ds.\n"  % (hours, mins, secs)
    print '------------------------------------------------------------'

    print ''
    print 'numFolds   : ', numFolds
    print 'numTrees   : ', numTrees
    print 'maxDepth   : ', maxDepth
    print 'maxBins    : ', maxBins
    print 'minInstancesPerNode: ', minInstancesPerNode
    print 'partition  : ', partition
    print 'cv model   : ', cvModel
    print 'best model : ', cvModel.bestModel
    print 'avg metric : ', cvModel.avgMetrics
    print '------------------------------------------------------------'


def predict(what='', 
            hdfreadfile='', 
            hdfwritefile='', 
            name='',
            isMC=False,
            isExp=False
            ):
    if (not isMC) and (not isExp):
        raise Exception("Provide either isMC=True or isExp=True.")
    hf           = tables.open_file(hdfreadfile, 'a')
    Pulses   = hf.root.Pulses[:]
    TankX    = hf.root.TankX[:]
    TankY    = hf.root.TankY[:]
    HitsTime = hf.root.HitsTime[:]
    if isMC:
        Nsta     = hf.root.Nsta[:]
        COGX      = hf.root.COGX[:]
        COGY      = hf.root.COGY[:]
        T0        = hf.root.T0[:]
        PlaneZenith  = hf.root.PlaneZenith[:]
        PlaneAzimuth = hf.root.PlaneAzimuth[:]

    elif opts.isExp:
        print 'isMC=False. isExp or isqgsjet'
        Nsta        = hf.root.Nstation.cols.value[:]
        COGX        = hf.root.ShowerCOG.cols.x[:]
        COGY        = hf.root.ShowerCOG.cols.y[:]
        PlaneZenith = hf.root.ShowerPlane.cols.zenith[:]
        PlaneAzimuth= hf.root.ShowerPlane.cols.azimuth[:]
        T0          = hf.root.ShowerCOG.cols.time[:]
        filterSTA2  = hf.root.IceTop_TwoStationFilter.cols.value[:]

    hf.close()

    partition = 420

    print ''
    what_list = ['x', 'y', 'zenith']
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
        
        predicted_var_name = 'PredictedX'
        rfPath = "/data/icet0/rkoirala/LowEnergy/RandomForest/RFcorexModel"+name
        
    # ====================================================================================  
    # Use RFModel on experimental data to get Y position.  
    elif what=='y':
        print 'Predicting Y'    
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
        
        predicted_var_name = 'PredictedY'
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
        
        predicted_var_name = 'PredictedZen'
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

# Now predict x, y, and zenith here.
if __name__ == "__main__":

    print 'Random Forest Regression with PySpark'
    print '-------------------------------------------------'
    print ''
    """
    To Run:
        python spark_predict_data_mc.py --isExp -f filename.hdf
    """
    from time import time
    start = time()

    # ========================================================================================
    # Import all packages required.
    import numpy, os, glob, tables
    from tables import IsDescription, Float64Col
    from pyspark import SparkContext, SparkConf
    from pyspark.sql.session import SparkSession
    from pyspark.sql import SQLContext
    import read # from h5spark package

    from pyspark.sql.types import * # FloatType, IntegerType, ArrayType etc
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.regression import RandomForestRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql.functions import *
    from pyspark.ml.linalg import Vectors, VectorUDT # save Pulses as vector so that it can be merged easily later for regression.
    from pyspark.sql.functions import rand           # Shuffle RDD randomly.
    #from pyspark.mllib.tree import RandomForest, RandomForestModel

    # Import all packages required.
    import numpy, os, glob, tables
    from pyspark import SparkContext, SparkConf
    from pyspark.sql.session import SparkSession
    from pyspark.sql import SQLContext
    import read # from h5spark package

    from pyspark.sql.types import * # FloatType, IntegerType, ArrayType etc
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.linalg import Vectors, VectorUDT # save Pulses as vector so that it can be merged easily later for regression.


    # =========== Nomenclature ============
    rfrdir       = '/data/icet0/rkoirala/LowEnergy/RandomForest/'
    isqgsjet     = False

    if isqgsjet:
        name_1sthalf = '_qgsjet_1sthalfMC'
        name_2ndhalf = '_qgsjet_2ndhalfMC'
        
        fn_1sthalf   = rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_qgsjet_1sthalf.h5'
        fn_2ndhalf   = rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_qgsjet_2ndhalf.h5'
    else:
        name_1sthalf = '_1sthalfMC'
        name_2ndhalf = '_2ndhalfMC'
        
        fn_1sthalf   = rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_1sthalf.h5'
        fn_2ndhalf   = rfrdir+'analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5'
    # ========Initialise spark session===============
    conf  = SparkConf()
    sc    = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Now train x, y, and zenith. Run one after another.   
    #train_x(hfreadfile=fn_1sthalf    , trial=False, name=name_1sthalf)
    #train_x(hfreadfile=fn_2ndhalf    , trial=False, name=name_2ndhalf)
    #train_y(hfreadfile=fn_1sthalf    , trial=False, name=name_1sthalf)
    #train_y(hfreadfile=fn_2ndhalf    , trial=False, name=name_2ndhalf)
    #train_theta(hfreadfile=fn_1sthalf, trial=False, name=name_1sthalf)
    #train_theta(hfreadfile=fn_2ndhalf, trial=False, name=name_2ndhalf)
    # Now predict x, y, and zenith. Use trained model from 1stfile and implement in 2ndfile and vice-versa.
    #predict(what='x', hdfreadfile=fn_1sthalf, hdfwritefile=fn_1sthalf, name=name_2ndhalf, isMC=True)
    #predict(what='x', hdfreadfile=fn_2ndhalf, hdfwritefile=fn_2ndhalf, name=name_1sthalf, isMC=True)
    #predict(what='y', hdfreadfile=fn_1sthalf, hdfwritefile=fn_1sthalf, name=name_2ndhalf, isMC=True)
    #predict(what='y', hdfreadfile=fn_2ndhalf, hdfwritefile=fn_2ndhalf, name=name_1sthalf, isMC=True)
    #predict(what='zenith', hdfreadfile=fn_1sthalf, hdfwritefile=fn_1sthalf, name=name_2ndhalf, isMC=True)
    predict(what='zenith', hdfreadfile=fn_2ndhalf, hdfwritefile=fn_2ndhalf, name=name_1sthalf, isMC=True)
    
    print ''
    total_time = time() - start    
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    print "Total running time: %dh:%dm:%ds.\n"  % (hours, mins, secs)
    print ' ---------------------------------------------------------------'
    
    spark.stop()
    # ============================DONE========================================================




    