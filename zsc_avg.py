#!/usr/bin/env python

#print 'Average distance to hit tanks w.r.t shower plane'
#print ''

import tables, numpy
from optparse import OptionParser

parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
parser.add_option('-f', '--filename', help="Do small shower recontruction.")
parser.add_option('--isMC', action='store_true')
parser.add_option('--isExp', action='store_true')
parser.add_option('--isqgsjet', action='store_true', help="is this simulation data?")
opts, args = parser.parse_args()

# ========================================================================================
# In shower coordinate system:
#    z direction: +ve z direction of shower is a perpendicular arrow from the shower plane 
#                 that goes through icetop towards in-ice.
#    x direction: x axis of shower is the line that is formed when plane shower font 
#                 touches the icetop geometry. +ve x-axis is towards the arrow that 
#                 is used to form azimuth (phi) angle in original IT geometry.
#    y direction: Cross product (right-hand rule) of +ve x and +ve y produces +ve z axis.
#                 Y-axis lies on the shower front.
def rotate_to_shower_cs(x,y,z,phi,theta,core_x,core_y,core_z):
    """ Input: X,Y,Z of the dom/particle; Phi, Theta, Core x,y ,z of the shower.                                                                             
        Output: Radial distance from shower core in shower coordinate system."""
    import numpy
    # counter-clockwise (pi + phi) rotation                                                                                                                  
    d_phi    = numpy.matrix([ [ -numpy.cos(phi), -numpy.sin(phi), 0],
                              [  numpy.sin(phi), -numpy.cos(phi), 0],
                              [  0,               0,              1]])
    # clock-wise (pi - theta) rotation                                                                                                                       
    d_theta  = numpy.matrix([ [  -numpy.cos(theta), 0, -numpy.sin(theta)],
                              [  0,                 1,  0,              ],
                              [  numpy.sin(theta),  0, -numpy.cos(theta)]])
    rotation = d_theta*d_phi

    origin   = numpy.array([[core_x], [core_y], [core_z]])
    
    det_cs_position    = numpy.array([[x],[y],[z]])
    shower_cs_position = rotation*(det_cs_position - origin)
    shower_cs_radius   = numpy.sqrt(shower_cs_position[0]**2 + shower_cs_position[1]**2)
    
    return shower_cs_position
    
# ========================================================================================
def zsc_avg(tankx, tanky, tankz, azimuth, zenith, corex, corey):
    # calculate the average z distance of hit tanks in shower coordinate system for one event.
    zsum  = 0
    nhits = 0
    for i in range(len(tankx)):
        if not ((tankx[i]==0) and (tanky[i]==0) and (tankz[i]==0)):
            nhits += 1.
            z      = rotate_to_shower_cs(tankx[i],tanky[i],tankz[i],azimuth,zenith,corex,corey,1946.)[2]
            zsum  += abs(z)
            
    return zsum/nhits

# ========================================================================================
# File option to choose from.
workdir         =   '/data/icet0/rkoirala/LowEnergy/RandomForest/'       
hf = tables.open_file(opts.filename)
TankX       = numpy.array(hf.root.TankX[:])
TankY       = numpy.array(hf.root.TankY[:])
TankZ       = numpy.array(hf.root.TankZ[:])                 
if opts.isMC:
    print 'isMC=True'
    #hfreadfile       = '/data/icet0/rkoirala/LowEnergy/analysis_simulation_sta2only_relativeTime_SLCHLC_background_Ntanks_leq85_allParticles.h5'
    #hfreadfile       = workdir+'corex_ml_final.h5'
    #hfreadfile       = workdir+'corex_mlnozencut.h5'
    #hfreadfile       = workdir+'corex_ml_162.h5'
    #hfreadfile       = workdir+'corex_ml_leq35.h5'
    #hfreadfile       = workdir+'analysis_simulation_sta2_Ntanks_leq35_iron.h5'
    # ========================================================================================
    # Get info from corex_ml.h5. It includes predicted X coordinate.
    # All these array has already been masked while running ML to get corex.
    COGX        = hf.root.COGX[:]
    COGY        = hf.root.COGX[:]
    PlaneZenith = hf.root.PlaneZenith[:]
    PlaneAzimuth= hf.root.PlaneAzimuth[:]
elif opts.isExp:
    print 'isMC=False'
    COGX        = hf.root.ShowerCOG.cols.x[:]
    COGY        = hf.root.ShowerCOG.cols.y[:]
    PlaneZenith = hf.root.ShowerPlane.cols.zenith[:]
    PlaneAzimuth= hf.root.ShowerPlane.cols.azimuth[:]
hf.close()

ZSC_avg = numpy.array([])
for i in range(len(TankX)):
    evt_tankx = TankX[i]
    evt_tanky = TankY[i]
    evt_tankz = TankZ[i]
    zscavg = zsc_avg(evt_tankx, evt_tanky, evt_tankz, PlaneAzimuth[i], PlaneZenith[i], COGX[i], COGY[i])
    ZSC_avg = numpy.append(ZSC_avg, zscavg)
    if (i%10000==0):
        print i, zscavg

hf = tables.open_file(opts.filename, 'a')
if 'ZSC_avg' in hf.root:
    hf.remove_node('/', 'ZSC_avg')
hf.create_array('/', 'ZSC_avg', ZSC_avg)
hf.close()

print 'ZSC DONE'
# ========================================================================================
