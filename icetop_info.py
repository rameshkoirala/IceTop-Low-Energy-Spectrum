#!/usr/bin/env python

import numpy, math
import matplotlib.pyplot as plt
params = {'legend.fontsize': 18,
          'axes.labelsize' : 22,
          'axes.titlesize' : 23,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'figure.figsize' : (10, 8),
          'axes.grid'      : True}
plt.rcParams.update(params)


from tanks_coordinate import all_coordinate

tank_positions = all_coordinate()


def distance_between_tanks(station=46):
    # --------------------------------------------
    # Calculate distance between two tanks of a station
    # --------------------------------------------     
    #tank_positions = self.it_geometry()
    posA = numpy.array(tank_positions[station]['A'])
    posB = numpy.array(tank_positions[station]['B'])
    distance = numpy.sqrt(sum((posA - posB)**2))
    print 'Distance between tanks in station %s :'%(station), distance, ' [m]'
    return distance

def distance_between_stations(station1=46, station2=81):
    # --------------------------------------------
    # Calculate distance between two stations
    # 1. Calculate mid point of tanks of a station
    # 2. Calculate distance between mid points of tanks
    #    of two stations                               
    # --------------------------------------------     
                                                      
    #tank_positions = self.it_geometry()
    sta1_posA = numpy.array(tank_positions[station1]['A'])
    sta1_posB = numpy.array(tank_positions[station1]['B'])
    sta2_posA = numpy.array(tank_positions[station2]['A'])
    sta2_posB = numpy.array(tank_positions[station2]['B'])
    
    mid1 = (sta1_posA + sta1_posB)/2.
    mid2 = (sta2_posA + sta2_posB)/2.
    distance = numpy.sqrt(sum((mid1 - mid2)**2))
    print "Distance between station %s and station %s is:"%(station1, station2), distance, '[m]'
    return distance

def plot_circle(radius, origin, color='r', label='R=200m', linewidth=3, alpha=1.):
    ax = plt.gca()
    ax.add_patch(plt.Circle(origin, radius,  color=color, linewidth=linewidth,
                     label=label, fill=False))
    ax.set_aspect('equal', adjustable='datalim')
    ax.plot()   #Causes an autoscale update.


def plot(plot_option='it_geometry', radius=410, origin=(0,0), color='r', **kwargs):
    
    # Plot IceTop Geometry which will act as the background for additional conditions.
    # XA, .... are array of positions of each tank of every stations of IT
    XA = numpy.array([tank_positions[station]['A'][0] for station in range(1,82)])
    YA = numpy.array([tank_positions[station]['A'][1] for station in range(1,82)]) 
    XB = numpy.array([tank_positions[station]['B'][0] for station in range(1,82)])
    YB = numpy.array([tank_positions[station]['B'][1] for station in range(1,82)])
    
    plt.scatter(XA, YA, color='b')
    plt.scatter(XB, YB, color='g')
    
    for i in range(81):
        if i+1 not in [26, 36, 46, 79, 80, 81]:
            plt.text(XA[i]+10, YA[i], i+1, fontsize=8)
        
    # Boundary outside IceTop Infill stations.
    if plot_option=='infill_boundary':
        X = numpy.array([-6.53999996, -9.11999989,  67.07500076, 
                          170.77000427, 154.95999908, -6.53999996])
        Y = numpy.array([-149.35499573, -87.69499969, 95.39500046, 
                          -11.06499958, -128.70999908, -149.35499573])
        plt.plot(X,Y, 'r', label="Boundary of Infill Stations")      

    # Boundary of Nearby Infill Stations
    elif plot_option=='nearby_infill_boundary':
        ka = numpy.array([-6.53999996, -9.11999989,  67.07500076, 125., 75., -6.53999996])
        kha = numpy.array([-149.35499573, -87.69499969, 95.39500046, 40., -140., -149.35499573])
        plt.plot(ka, kha, 'b', label='Boundary of Nearby Infill Stations', linewidth=3)
        plt.legend(loc=2,prop={'size':8})

    # Outside Boundary with LaputopFractionContainment == 1.
    # Boundary stations are [1, 6, 50, 74, 72, 78, 75, 31]  
    elif plot_option=='outside_boundary':
        x = numpy.array([-260.61499786, 361.40750122, 600.44999695, 365.70500183, 120.08250237,
                          4.49999869e-02, -356.92499542, -550.71749878, -260.61499786])
        y = numpy.array([-496.98249817, -398.25500488, 144.05749893, 428.20500183, 397.2124939,
                          4.99107498e+02, 422.79750824, -145.46749878, -496.98249817])
        plt.plot(x,y, 'k', label="IT Boundary")
        plt.legend(loc=2,prop={'size':8})

    elif plot_option=='circle':
        plot_circle(radius=radius, origin=origin, **kwargs)
        plt.legend(loc=2,prop={'size':8})
      
    elif plot_option=='it_geometry':
        pass

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.xlim(-800,800)
    plt.ylim(-800,800)
    plt.title("IceTop Geometry")
    plt.show()

def plot_trigger_circle():

    XA = numpy.array([tank_positions[station]['A'][0] for station in range(1,82)])
    YA = numpy.array([tank_positions[station]['A'][1] for station in range(1,82)]) 
    XB = numpy.array([tank_positions[station]['B'][0] for station in range(1,82)])
    YB = numpy.array([tank_positions[station]['B'][1] for station in range(1,82)])

    station_list = [26, 36, 46, 79, 80, 81]
    radius = 60
    plt.figure(figsize=(10,8))
    ax = plt.gca()
    for station in station_list:

        if station==26:
            plt.text(XA[station-1]+13, YA[station-1]-5, station, fontsize=18)
        elif station==79:
            plt.text(XA[station-1]+8, YA[station-1]-10, station, fontsize=18)
        elif station==46:
            plt.text(XA[station-1]+10, YA[station-1], station, fontsize=18)
        elif station==80:
            plt.text(XA[station-1]-10, YA[station-1]+10, station, fontsize=18)
        elif station==81:
            plt.text(XA[station-1]-25, YA[station-1], station, fontsize=18)
        elif station==36:
            plt.text(XA[station-1]+10, YA[station-1]-5, station, fontsize=18)
        elif XA[station-1]<320. and XA[station-1]>-250. and abs(YA[station-1])<300.:
            plt.text(XA[station-1], YA[station-1]-20, station, fontsize=18)

        #plot_circle(radius, (XA[station-1], YA[station-1]))
        #plot_circle(radius, (XB[station-1], YB[station-1]))
        
        ax.add_patch(plt.Circle((XA[station-1], YA[station-1]), radius,  linewidth=2.5, color='r', fill=False))
        ax.add_patch(plt.Circle((XB[station-1], YB[station-1]), radius,  linewidth=2.5, color='r', fill=False))
        ax.set_aspect('equal', adjustable='datalim')
        plt.scatter(XA[station-1], YA[station-1], color='k', s=100, alpha=0.6)
        plt.scatter(XB[station-1], YB[station-1], color='gray', s=100, alpha=0.6)

    ka = numpy.array([-25, -25,  67, 135., 85., -25])     # These are hand picked values to locate infill stations.
    kha = numpy.array([-160, -88, 140, 35., -170., -160]) # These are hand picked values to locate infill stations.
    #plt.plot(ka, kha, 'k', label='Infill Boundary', linewidth=3, alpha=0.2)
    plt.grid(ls='--', alpha=0.5)
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('IceTop Infill Volume Trigger [Radius = 60 m]')
    #plt.savefig('icetop_infill_60m.png', bbox_inches='tight')
    plt.show()

plot_trigger_circle()