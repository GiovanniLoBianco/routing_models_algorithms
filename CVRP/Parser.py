#!/usr/bin/env python
# coding: utf-8

import numpy as np

# Generate instance of file f.
# fsol is the solution file corresponding to instance f --> we need it to get Vmax and m.
# slackVmax is the percentage (between 0 and 1) of allowed visits for one vehicle, 
# regarding the actual maximum number of visits in the solution.
# If set to 0, then Vmax will be exactly the max nb of visit in the solution fsol.
# If set to > 0, then the problem is more flexible, but the search space is larger too.
def parse(f, fsol, slackVmax):
    
    instance = dict()
    stopWhile = False
    
    with open(f, 'r') as file:
        line = "start"
        while not stopWhile:
            line = file.readline()
            w = line.split()
            if w[0]=='NAME':
                instance['name'] = w[-1]
            if w[0]=='DIMENSION':
                instance['n'] = int(w[-1])-1
            if w[0]=='CAPACITY':
                instance['capacity'] = int(w[-1])
            if w[0]=='NODE_COORD_SECTION':
                stopWhile = True
                line = file.readline()
                ww = line.split()
                x_d = float(ww[1])
                y_d = float(ww[2])
                coordinates= [(x_d,y_d)]
                while line.split()[0] != "DEMAND_SECTION":
                    line = file.readline()
                    ww = line.split()
                    if len(ww)==3:
                        coordinates.append((float(ww[1]), float(ww[2])))
                n = instance['n']
                traveltimes = np.zeros((n+1,n+1), np.float32)
                for i in range(n):
                    (x_i, y_i) = coordinates[i] 
                    for j in range(i+1, n+1):
                        (x_j, y_j) = coordinates[j]
                        traveltimes[i][j] = np.sqrt((x_j-x_i)**2 + (y_j-y_i)**2)
                        traveltimes[j][i] = traveltimes[i][j]
                instance['coordinates'] = coordinates
                instance['times'] = traveltimes
                demands = []
                while line.split()[0] != "DEPOT_SECTION":
                    line = file.readline()
                    ww = line.split()
                    if len(ww)==2:
                        demands.append(int(ww[-1]))
                instance['demands'] = demands[1:]
                
    # Generate angle of cities around the depot
    angles = np.zeros(n, np.float32)
    for i in range(1, n+1):
        (x_i,y_i) = coordinates[i]
        angles[i-1] = np.arctan2(y_i-y_d, x_i-x_d)
    instance['angles'] = angles
    
    # Generate angle gaps between each pair of city, excluding depot
    angleGap = np.zeros((n,n), np.float32)
    for i in range(n):
        for j in range(i+1,n):
            angleGap[i][j] = min(np.abs(angles[j]-angles[i]), 2*np.pi-np.abs(angles[j]-angles[i]))
            angleGap[j][i] = angleGap[i][j]
    instance['angleGap'] = angleGap
    
    
    # To get m and Vmax, we need to read to the solution file
    with open(fsol, 'r') as solFile:
        stopWhile = False
        m = 0
        Vmax = 0
        while not stopWhile:
            line = solFile.readline()
            w = line.split()
            if w[0] != "Cost":
                m += 1
                Vmax = max(Vmax, len(w)-2)
            else:
                stopWhile = True
        instance['m'] = m
        instance['Vmax'] = (int)(np.ceil(Vmax*(1 + slackVmax)))

    
    d = sorted(instance["demands"])
    idx = 0
    cumul = 0
    while d[idx] + cumul <= instance['capacity']:
        cumul += d[idx]
        idx += 1
    instance['Vmax'] = idx
      
    return instance
