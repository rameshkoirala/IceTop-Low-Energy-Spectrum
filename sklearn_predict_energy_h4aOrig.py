#!/usr/bin/env python

def feature_importance_dict(permutation_feature, featuresname, n=35, nmodels=3):
    permutation_feature[permutation_feature<0]=0
    permutation_feature = permutation_feature/sum(permutation_feature)
    
    feature_importance = {}
    for feature in featuresname:
        feature_importance[feature] = []

    i = 0
    # feature_importance = {'lNtanks':[val1, val2, val3, ..val_kfold], 'lQhillas':[....], ...}
    for feature in featuresname:
        for j in range(nmodels):
            if feature=='pulses':
                feature_importance[feature].append(sum(permutation_feature[i:i+n]))
            elif feature=='rhit':
                feature_importance[feature].append(sum(permutation_feature[i:i+n]))
            else:
                feature_importance[feature].append(permutation_feature[i])

        if feature=='pulses':
            i += n
        elif feature=='rhit':
            i += n
        else:
            i += 1

    return feature_importance

def mean_feature_importance(fi_dict, 
                            savehdf=False,
                            composition_model='',
                            particle='',
                            name=''
                           ):
    # Average Feature Importance
    # fi_dict has dict of fi_name and list of values for each k-folds.
    mean_fi = {}
    for (fn, val) in fi_dict.items():
        mean_fi[fn] = numpy.mean(val)

    mean_fi = sorted(mean_fi.items(), key=lambda (k, v): v, reverse=True) 

    featname = []
    featval  = []
    for item in mean_fi:
        featname.append(item[0])
        featval.append(item[1])

    if savehdf:
        class Particle(IsDescription):
            logqtotal   = Float64Col()
            logqsum2    = Float64Col()
            lognstation = Float64Col()
            costheta    = Float64Col()
            pulses      = Float64Col()
            rhit        = Float64Col()

        hf = tables.open_file(hfreadfile, 'a')
        if 'featureImportancesEnergy_'+composition_model in hf.root:
            hf.remove_node('/', 'featureImportancesEnergy_'+composition_model)
        table = hf.create_table('/', 'featureImportancesEnergy_'+composition_model, Particle, "feature importances for Energy")
        particle = table.row
        particle[featname[0]] = featval[0]
        particle[featname[1]] = featval[1]
        particle[featname[2]] = featval[2]
        particle[featname[3]] = featval[3]
        particle[featname[4]] = featval[4]
        particle[featname[5]] = featval[5]
        particle.append()
        table.flush()
        hf.close()

    
    print ''
    print 'feature Name      :', featname
    print 'feature importance:', featval

    return mean_fi

def permutation_importances_raw(rf, X_train, y_train, metric):
    """
    Return array of importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    """
    import numpy as np
    
    baseline = metric(rf, X_train, y_train)
    X_train_orig = X_train.copy()
    imp = []
    for i in range(X_train.shape[1]):
        save = (X_train[:,i]).copy()
        X_train[:,i] = np.random.permutation((X_train[:,i]).copy())
        m = metric(rf, X_train, y_train)
        X_train = X_train_orig.copy()
        imp.append(baseline - m)
        
    return np.array(imp)

def oob_regression_r2_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    # Copied from rfpimp.py
    # https://github.com/parrt/random-forest-importances/blob/master/src/rfpimp.py
    from sklearn.ensemble.forest import _generate_unsampled_indices
    from sklearn.metrics import r2_score
    import numpy as np
    
    X = X_train
    y = y_train

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds        = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices]   += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions

    oob_score = r2_score(y, predictions)
    return oob_score

def column_stack_block(filename, composition_model='h4a', name=''):
    # DO NOT use 'column_stack' as function name because it matches with 'numpy.colum_stack'.
    import tables, numpy
    
    print 'Reading data from: ', filename
    if isMC:
        hf = tables.open_file(filename)
        Tenergy    = hf.root.Energy[:]
        FiltCondt  = hf.root.FiltCondt[:]
        LoudSta    = hf.root.LoudSta[:]
        Qtotalhlc  = hf.root.Qtotalhlc[:]
        Nsta       = hf.root.Nsta[:]
        PredictedX = hf.root.PredictedX[:]
        PredictedY = hf.root.PredictedY[:]
        Pzenith    = hf.root.PredictedZen[:]
        Pulses     = hf.root.Pulses[:]
        Qsum2      = numpy.sum(Pulses[:,:2], axis=1)
        Qhillas    = numpy.sum(Pulses[:,1:], axis=1)
        TankX      = hf.root.TankX[:]
        TankY      = hf.root.TankY[:]
        Weight     = hf.root.WeightH4aOrig[:]
        PDG        = hf.root.pdg_encoding[:]
        hf.close()
        
        #----------------------------
        # Calculated 
        Radius     = numpy.sqrt((TankX - PredictedX[:,numpy.newaxis])**2 + (TankY - PredictedY[:,numpy.newaxis])**2)
        Radius[numpy.where((TankX==0)*(TankY==0))] = 0. #If nothing was hit, tank positions was set to (0,0) to fill array. 
        Radius     = -numpy.sort(-Radius, axis=1)       # Distance of tank-core in descending order per event.
        fracRadius = Radius/Rref

        # Quality Cuts so that you train on nicely behaving events.
        qualmask  = FiltCondt==1
        qualmask *= (LoudSta==1)
        qualmask *= (Qsum2<=qsum2_frac*Qtotalhlc) # Make sure not all charge is stored in 1 station.
        qualmask *= (numpy.cos(Pzenith)>=0.8)*(numpy.cos(Pzenith)<=1.0)
    
        hf = tables.open_file(filename, 'a')
        if 'maskTrainingEnergy_'+composition_model+name in hf.root:
            hf.remove_node('/', 'maskTrainingEnergy_'+composition_model+name)
        hf.create_array('/', 'maskTrainingEnergy_'+composition_model+name, qualmask)
        hf.close()
    
        print 'Events after all quality cuts: ', sum(qualmask)
    
        # Apply mask on arrays before training
        Tenergy    = Tenergy[qualmask]
        Pzenith    = Pzenith[qualmask]
        Qtotalhlc  = Qtotalhlc[qualmask]
        Qsum2      = Qsum2[qualmask]
        Qhillas    = Qhillas[qualmask]
        Pulses     = Pulses[qualmask]
        fracRadius = fracRadius[qualmask]
        PredictedX = PredictedX[qualmask]
        PredictedY = PredictedY[qualmask]
        Nsta       = Nsta[qualmask]
        Weight     = Weight[qualmask]
    
        # === order of featuresname matter ====
        featuresname = ['logqtotal', 'logqsum2', 'lognstation', 'costheta', 'pulses','rhit']
        col_stack = numpy.column_stack((numpy.log10(Tenergy),
                                        numpy.log10(Qtotalhlc),
                                        numpy.log10(Qsum2),
                                        #numpy.log10(Qhillas),
                                        numpy.log10(Nsta),
                                        numpy.cos(Pzenith),
                                        numpy.log10(Pulses),
                                        numpy.log10(fracRadius),
                                        Weight
                                       ))


    elif isExp:
        # Predict energy for everything and remove events outside of quality cuts later.
        hf = tables.open_file(filename)
        Qtotalhlc  = hf.root.TotalChargeHLC.cols.value[:]
        Nsta       = hf.root.Nstation.cols.value[:]
        # Exactly mimic what has been done in MC. Keep track of what trained model is used
        # to predict X, Y, and Zenith.
        if name=='_1sthalfMC':
            usepredicted = '2nd'
        elif name=='_2ndhalfMC':
            usepredicted = '1st'
        elif name=='_QGSJET':
            usepredicted = '_qgsjet'
        else:
            usepredicted = ''
            
        PredictedX = hf.get_node('/', 'PredictedX'+usepredicted)[:]
        PredictedY = hf.get_node('/', 'PredictedY'+usepredicted)[:]
        Pzenith    = hf.get_node('/', 'PredictedZen'+usepredicted)[:]
        Pulses     = hf.root.Pulses[:]
        Qsum2      = numpy.sum(Pulses[:, :2], axis=1) # sum of two highest Pulses.
        TankX      = hf.root.TankX[:]
        TankY      = hf.root.TankY[:]
        hf.close()
        
        #----------------------------
        # Calculated 
        Radius     = numpy.sqrt((TankX - PredictedX[:,numpy.newaxis])**2 + (TankY - PredictedY[:,numpy.newaxis])**2)
        Radius[numpy.where((TankX==0)*(TankY==0))] = 0. #If nothing was hit, tank positions was set to (0,0) to fill array. 
        Radius     = -numpy.sort(-Radius, axis=1)       # Distance of tank-core in descending order per event.
        fracRadius = Radius/Rref
    
        featuresname = ['logqtotal', 'logqsum2', 'lognstation', 'costheta', 'pulses','rhit']
        col_stack = numpy.column_stack((numpy.log10(Qtotalhlc),
                                        numpy.log10(Qsum2),
                                        numpy.log10(Nsta),
                                        numpy.cos(Pzenith),
                                        numpy.log10(Pulses),
                                        numpy.log10(fracRadius),
                                       ))                                       
    
    return col_stack, featuresname

def predict_energy(column_stack, 
                   composition_model =''   ,
                   particle          ='all',
                   use_saved_model   =True ,
                   max_features      ='auto',
                   numTrees          =600  ,
                   maxDepth          =10   ,
                   minInstancePerNode=10   ,
                   n_splits          =3    ,
                   name              =''   ,
                   featuresname = ['lQtot', 'lQsum2', 'lNsta', 'coszen', 'lQhit','lRhit']
                  ):
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
 
    skf                = KFold(n_splits=n_splits)
    models             = []
    test_samples       = []
    scores             = []
    permut_fi_list     = []

    print 'Predicting energy for model: %s'%composition_model
    
    # Experimental data does not have Tenergy and can never be used for training. It is 
    #  only used for prediction.
    if isMC:
        label     = column_stack[:,0]
        features  = column_stack[:,1:-1]
    if isExp:
        features  = column_stack
    
    features  = features.astype(numpy.float32)
    features  = numpy.nan_to_num(features)

    rfr_model_filename = thisdir+'rfr_finalized_model_all_'+composition_model+name+'.sav'
    test_samples_file  = thisdir+'test_samples_all_'+composition_model+name+'.sav'
    print "Model: ", rfr_model_filename
    
    # Use saved model
    if use_saved_model:
        print 'Using saved Model %s.'%rfr_model_filename
        models       = pickle.load(open(rfr_model_filename, 'rb'))
        tsfile       = open(test_samples_file,'rb')
        test_samples = pickle.load(tsfile)
        tsfile.close()

    # Train MC data to generate Models to predict energy.
    elif (not use_saved_model) and (not isExp):
        print 'Training and creating Model %s.'%rfr_model_filename
        weight  = column_stack[:,-1]
        models  = []     
        feature_importance = {}
        for feature in featuresname:
            feature_importance[feature] = []

        for train, test in skf.split(features, label):
            print "================================="
            print "build new RF ... "
            '''
            rfr = RandomForestRegressor(n_jobs=-1, oob_score = True)
            clf = GridSearchCV(estimator=rfr, param_grid=param_grid)
            '''
            rfr = RandomForestRegressor(n_estimators=numTrees, 
                                        n_jobs=-1,
                                        max_depth=maxDepth,
                                        min_samples_leaf=minInstancePerNode
                                       )
            print "done."
            print "Training on ", train
            models.append(rfr.fit(features[train], label[train], sample_weight=weight[train]))
            imp     = permutation_importances_raw(rfr, features[train], label[train], oob_regression_r2_score)
            fi_dict = feature_importance_dict(imp, featuresname, nmodels=n_splits) # function that returns featurename and its value.
            for key,val in fi_dict.items():
                feature_importance[key].append(val[0])
            print "append test sample", test, " to test samples"
            print "Feature Importance", fi_dict
            test_samples.append(test)
            del rfr

        if savemodel: #defined at steering file
            # Save models and list of test samples.
            pickle.dump(models, open(rfr_model_filename, 'wb'))
            filehandler = open(test_samples_file,"wb")
            pickle.dump(test_samples,filehandler)
            filehandler.close()
        if savehdf:  #defined at steering file
            # save featurename and feature importance value in descending order.
            mean_feature_importance(feature_importance, 
                                    savehdf=savehdf,
                                    composition_model=composition_model,
                                    particle=particle,
                                    name=name
                                   )

    else:
        raise Exception("If you want to train models for energy prediction, make sure you \
                         are not giving experimental data. Use option --isMC.")
        
    if isExp or isqgsjet:
        avg_predicted = 0.
        for i in range(len(models)):
            print "================================="
            print "using model with index ", i
            avg_predicted += models[i].predict(features)
            
        penergy = avg_predicted/(1.*len(models))
        
    elif isMC:
        predicted = numpy.array([])
        for i, test in enumerate(test_samples):
            print "================================="
            print "using model with index ", i
            prediction = models[i].predict(features[test])
            predicted = numpy.append(predicted, prediction)
            print "append prediction of shape", prediction.shape, "to array"
        
        penergy = predicted

    else:
        raise Exception("Tell me if this is MC, Exp, or QGSJET data. Enter --isMC or --isExp or --isqgsjet")  
        
    print "================================="
    return penergy

if __name__ == "__main__":
    '''
    This script is used to predict energy of events from experimental data.
    First create a model from simulation with Sibyll2.1 and QGSJetII-04 and save it.
    Then use the saved model to predict energy of events from experimental data.
    Experimental data is huge, one per Run number. Use 'sklearn_predict_energy_exp_create_submit.py'
        to submit one job per Run number in Asterix cluster. 

    To Run:
        python sklearn_predict_energy_h4aOrig.py --savemodel --isMC --savehdf --do predict -f analysis_simulation_HLCCoreSeed_slcQcut_fracradius_final.h5
        python sklearn_predict_energy_h4aOrig.py --savemodel --isMC --isqgsjet --savehdf --do predict -f analysis_simulation_HLCCoreSeed_slcQcut_everything_qgsjet_final.h5
        python sklearn_predict_energy_h4aOrig.py --isExp --savehdf --usesavedmodel --do predict -f {hfname} # done by 'sklearn_predict_energy_exp_create_submit.py'
    
    Saved Models:
        rfr_finalized_model_all_h4aOrig.sav
        rfr_finalized_model_all_h4aOrig_QGSJET.sav
    '''

    import numpy, tables
    import pickle
    from tables import *
    from time import time
    start = time()

    from optparse import OptionParser
    parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
    parser.add_option('-f', '--filename', default='', help="Do energy recontruction.")
    parser.add_option('--do', default='all', help="Do small shower recontruction.")
    parser.add_option('--savemodel', action='store_true', help="Save recently trained RFR model")
    parser.add_option('--usesavedmodel', action='store_true', help="Used already trained saved RFR model")
    parser.add_option('--savehdf', action='store_true', help="Save hdf files")
    parser.add_option('--isExp', action='store_true', help="is this experimental data?")
    parser.add_option('--isMC', action='store_true', help="is this simulation data?")
    parser.add_option('--trial', action='store_true', help="run a faster trial version")
    parser.add_option('--isqgsjet', action='store_true', help="is this QGSJET simulation data?")

    opts, args = parser.parse_args()

    name               = 'Orig' #_1sthalfMC , _2ndhalfMC, _halfMC, _noLog, _hiX, _mnQ, _radiusCS, _QGSJET
    particle           = 'all'    # 'proton', 'helium', 'oxygen', 'iron', 'all'
    composition_model  = 'h4a'    # 'h4a', 'gst', 'poly', 'gsfL', 'gsfM', 'gsfH'
    # Values
    cos1               = 1.00   # upper bound: 1.0
    cos2               = 0.80   # Lower bound: 0.9
    qsum2_frac         = 0.90   # Qsum2<=qsum2_frac*Qtotal
    Rref               = 60.0   # fracRadius = Radius/Rref
    # Boolean
    isqgsjet           = opts.isqgsjet
    savehdf            = opts.savehdf  # save important results, like flux, Aeff, resolution etc, in hdf files
    savemodel          = opts.savemodel
    use_saved_model    = opts.usesavedmodel  # Use saved RandomForest Regression model rather than re-running it.
    isMC               = opts.isMC
    isExp              = opts.isExp
    
    datadir = "/data/icet0/rkoirala/LowEnergy/RandomForest/"
    thisdir = "/data/icet0/rkoirala/LowEnergy/RandomForest/"

    if isqgsjet:
        namet += '_QGSJET'

    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # Give input hdffile which is used to create column stack.
    # qgsjet: datadir+'analysis_simulation_sta2_Ntanks_leq35_qgsjet_all.h5'
    # else  : datadir+'analysis_simulation_HLCCoreSeed_slcQcut_2ndhalf.h5'
    hfreadfile = opts.filename
    if opts.filename=='':
        raise Exception("Please provide proper input filename. You provided %s"%hfreadfile)

    # Start Predicting here.
    # Create necessary column stack
    col_stack, featuresname= column_stack_block(hfreadfile, composition_model=composition_model, name=name)
    if opts.trial:
        col_stack = col_stack[:1000]
    # Predict
    penergy = predict_energy(col_stack, 
                             composition_model=composition_model,
                             use_saved_model  =use_saved_model,
                             name             =name,
                             featuresname     =featuresname
                            )
    
    if savehdf:
        tempname = name
        hf = tables.open_file(hfreadfile, 'a')
        if 'PredictedLogEnergy_'+composition_model+tempname in hf.root:
            hf.remove_node('/', 'PredictedLogEnergy_'+composition_model+tempname)
        hf.create_array('/', 'PredictedLogEnergy_'+composition_model+tempname, penergy)
        hf.close()
            
    total_time = time() - start    
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    print "Total running time: %dh:%dm:%ds.\n"  % (hours, mins, secs)
    print '-------------------------END-----------------------------------'
