#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import sys

import QAP
import TSP

## First cluster cities together then solve a TSP on each cluster.
## cost is angle gap or traveled times or combination of them for the clustering part.
## tlABC and tlTSP are the time limits for the clustering method and for each TSP run call.
def ctsp(instance, cost, tlC, tlTSP, printout=False):
    
    ## Get data from instance
    times = instance['times']
 
    if printout:
        print("Clusterizing...")
    clusters, _ = QAP.solve(instance, cost, tlC)
    
    # Initialize solution to empty solution
    routes=[]
    lengthRoutes = []
    obj = 0

    if printout:
        print("Solving TSP...")
    for c in range(len(clusters)):
        if printout:
            print("...on cluster "+str(c+1)+"/"+str(len(clusters)))
        
        # For each cluster, we add the depot node and solve TSP
        cluster = [0]+clusters[c]
        subTimes = [[times[i][j] for j in cluster] for i in cluster]
        route, length, _, _ = TSP.solveWithGRB(len(cluster), subTimes, tlTSP)
        
        # Get true indices of cities from original problem and add the route to the solution
        trueRoute = [cluster[route[city]] for city in range(len(route))]
        routes.append(trueRoute)
        lengthRoutes.append(length)
        obj+=length
        
    return routes, obj, lengthRoutes


## Angle based clustering + TSP
def abctsp(instance, tlABC, tlTSP, printout=False, accAngle = 3):
    n = instance['n']
    cost = [[np.round(instance['angleGap'][i][j] * 10**accAngle, 0) for j in range(n)] for i in range(n)]
    return ctsp(instance, cost, tlABC, tlTSP, printout)


## Traveled times based clustering + TSP --> Clustering has to be with DA
def tbctsp(instance, tlTBC, tlTSP, printout=False, accTime = 3):
    n = instance['n']
    cost = [[np.round(instance['times'][i][j] * 10**accTime, 0) for j in range(1, n+1)] for i in range(1, n+1)]
    return ctsp(instance, cost, tlTBC, tlTSP, printout)


## Combination of angle gaps and traveled times, weighted by alpha: alpha*angleGap + (1-alpha)*time
## We first normalize each kind of cost regarding max angle gap and max time
## Then we compute the cost for each pair of cities, with accuracy acc.
## And run ctsp with this cost --> we need to use DA for clustering
def atbctsp(instance, alpha, tlATBC, tlTSP, printout = False, acc=3):
    n = instance['n']
    maxAngleGap = np.max(instance['angleGap'])
    maxTimes = np.max(instance['times'])
    cost = [[ np.round(alpha*(instance['angleGap'][i][j]/maxAngleGap)*10**acc + (1-alpha)*(instance['times'][i+1][j+1]/maxTimes)*10**acc,0)  for j in range(n)] for i in range(n)]
    return ctsp(instance, cost, tlATBC, tlTSP, printout)







