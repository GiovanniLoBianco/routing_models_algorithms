#!/usr/bin/env python
# coding: utf-8

import numpy as np

from GeneratePDPInstances import generateTimes


def parse(f):
    
    """
        Generate PDP instance of file f. Do no consider any temporal data.
        
        slackVmax is the percentage (between 0 and 1) of allowed visits for one vehicle: 
        
                    Vmax = ceil(m/(2*n) * 2)
        
    """
    
    # Initialize instance
    instance = dict()
    instance["name"] = f.split("/")[-1][:-4]
    
    # Open file and read all lines
    file = open(f, "r")
    lines = file.readlines()
    
    # Get nb. of requests
    n = int((len(lines)-2)/2)  
    instance["n"] = n
    
    # Get nb. of vehicles and capacity
    w0 = lines[0].split()
    instance["m"] = int(np.ceil(2*n/10)) # arbitrary, nb of vehicle available if instances are too large
    instance["capacity"] = int(w0[1])
    
    # For each tasks (pickup and delivery)
    # We get demands and coordinates for each request
    # Tasks are not ordered such that pickup task i corresponds to delivery task i+n
    # But each pickup task points to its delivery task in the data
    
    coordinates = [0 for i in range(2*n)]
    demands = [0 for i in range(2*n)]   
    i = 0
    for idx in range(2, len(lines)):
        w = lines[idx].split()
        
        idx_deli = int(w[-1]) # index of the delivery task associated to i if i is a pickup task
        if idx_deli != 0: # if this is a pickup task
            coordinates[i] = (int(w[1]), int(w[2])) # get coordinates
            demands[i] = int(w[3]) # get demand for pickup location
            
            w = lines[idx_deli+1].split() # get data of corresponding delivery location (+1 because first line is not a task)
            coordinates[i+n] = (int(w[1]), int(w[2])) # get coordinates
            demands[i+n] = int(w[3]) # get demand (= -1*demand of pickup)
            
            i+=1
            
    instance["coordinates"] = coordinates
    instance["demands"] = demands
    
    # Compute traveling times from coordinates
    instance["times"] = generateTimes(coordinates)
    
    # Compute Vmax
    instance["Vmax"] = int(np.ceil(np.ceil(2*n/instance["m"]) * 2))
    

    return instance
