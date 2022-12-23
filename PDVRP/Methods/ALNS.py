#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import os, sys

import InsertionHeuristics as insert
import RemovalHeuristics as remove

import copy
from sortedcontainers import SortedList


def initialize_regret2(instance):
    """
        Build a solution by inserting every request with regret-2 heuristic.
    """
    
    n = instance["n"]
    
    routes = []
    lengthRoutes = []
    requests = [i for i in range(n)]
    reqAssignment = [0 for i in range(n)]
    
    insert.regret(instance, routes, lengthRoutes, requests, 2, reqAssignment)
    
    return routes, lengthRoutes, reqAssignment  


def _wasAccepted(hObj, hlR, hSol, obj, lR, sol):
    
    """
        Routine to check if a solution has been registered already in hash tables.
        We first check if the objective has been found before. If yes, then we check the route's length.
        If list of routes' length match one of the accepted solution, then we check actual routes.
        The method works even if routes and list of route's length are not in the same order.
        
        To make the search even faster, hObj is a sorted list and the search of route's length is
        limited to the solution with same objective as obj. Idem for search of routes.
        
        hObj, hlR and hSol are such that hSol[i], hlR[i] and hObj[i] correspond to the same solution.
        
        Most of the time, a new accepted solution will have a different objective value, and even if this
        objective value has already been found before, it is very likely that the routes' length will be
        different. It should be very rare to check the actual sequences of cities visited.
        
        We use this method as hash function to determined wether or not a solution was already visited before.
        
    """
    
    # First check if the objective of the new solution has already been found
    if obj in hObj:
        
        # If yes, we check if the new route's lengths are not already in hlR
        # But we limit our seach to the accepted solution with objective value = obj
        idxFirstObj = hObj.index(obj)
        idxLastObj = idxFirstObj + hObj.count(obj)
        
        # Store indices for which lengths of routes are the same, initialized with all index of solution
        # with objective value = obj
        idx_identical = [idx for idx in range(idxFirstObj,idxLastObj)] 
        for i in range(len(idx_identical)-1, -1, -1): # backward loop does not change index ordering when removing element
            idx = idx_identical[i]
            lR_acc = hlR[idx]
            for l in lR:
                if l not in lR_acc:
                    del idx_identical[i] # We remove index from the list if routes' length are not identical
                    break
        
        # If there are still some indices in idx_identical, we need to check if at least one of the
        # corresponding accepted solutions are identical to the new solution.
        if len(idx_identical) > 0:
            for i in range(len(idx_identical)-1, -1, -1):
                idx = idx_identical[i]
                sol_acc = hSol[idx]
                for r in sol:
                    
                    # If we find at least one different route, 
                    # then we check the next accepted solution from idx_identical
                    if r not in sol_acc:
                        del idx_identical[i] 
                        break
                        
                    # If we reach this line, then we found a match of the new solution
                    # in the already accepted solution.
                    if r == sol[-1]:                      
                        return True        
            return False     
        else:
            return False
    else:
        return False


def solve(instance, initSol, maxIter, printout=False, lightprintout = True): 
    
    """
        Run ALNS such as described in
        "An Adaptive Large Neighborhood Search Heuristics for the Pick-Up and Delivery Problems with Time Windows",
        S. Ropke and D. Pisinger, 2006.
        
        - instance: instance to solve.
        
        - initSol: initial solution:
            initSol["routes"] : list of routes of initial solution.
            initSol["obj"] : objective value associated with first solution
            initSol["lengthRoute"] : distance travaled for each route, in the same order as in initSol["routes"]
            
        - maxIter: maximum number of iterations. 
        
    """
    
    starttime = time.time()
    
    ## ALNS Parameters (values after tuning, see XP part of the paper)
    phi = 9 # relatedness computation (distance)
    psi = 2 # relatedness computation (load)
    
    p = 6 # randomness in Shaw removal heuristic
    p_worst = 3 # randomness in Worst removal heuristic
    
    w = 0.05 # start temperature control parameter
    c = 0.99975 # cooling rate
    
    sigma = [33,9,13] # score adjustment parameters
    r = 0.1 # reaction factor
    
    eta = 0.025 # noise rate
    epsilon = 0.4 # neighborhood size
    
    lengthSegment = 100
    
    
    # Initialization of score, weight and selection probability for each heuristic
    # Rem: 0 --> worst, 1 --> random, 2 --> shaw
    # Ins: 0 --> greedy, 1 --> regret-2, 2 --> regret-3, 3 --> regret-4, 4 --> regret-m
    # Noise: 0 --> insertion cost without noise, 1 --> with noise
    
    scoreRem = [0,0,0]
    scoreIns = [0,0,0,0,0]
    scoreNoise = [0,0]
    
    nbUsedRem = [0,0,0]
    nbUsedIns = [0,0,0,0,0]
    nbUsedNoise = [0,0]
    
    weightRem = [0,0,0]
    weightIns = [0,0,0,0,0]
    weightNoise = [0,0]
    
    probaRem = [1/3, 1/3, 1/3]
    probaIns = [1/5,1/5,1/5,1/5,1/5]
    probaNoise = [1/2, 1/2]
    
    
    # First solution
    routes = initSol["routes"]
    obj = initSol["obj"]
    lengthRoutes = initSol["lengthRoute"]
    reqAssignment = [-1 for i in range(instance["n"])]
    for k, route_ini in enumerate(routes):
        for req in route_ini:
            if req < instance["n"]:
                reqAssignment[req] = k
    
    if printout or lightprintout:
        print("First solution: ", obj)
    
    # Best solution over all visited solutions
    bestObj = obj
    bestRoutes = copy.deepcopy(routes)
    
    histo_obj = [obj] # store best objective value found so far at each iteration
    obj_time = [(obj, 0)] # store objective and time every time a new best solution is found
    
    # Hashing function
    hashObj = SortedList() # store every objective value of accepted solution, sorted
    hashLR = [] # store list of route's length of every accepted solution, sorted according to hashObj
    hashSol = [] # store every accpeted solution, sorted according to hashObj
    # An accepted solution hashSolution[s] has route's length hashLength and objective value hashObj[s]
    
    # We add first solution to the hash tables
    hashObj.add(obj)
    hashLR.append(copy.deepcopy(lengthRoutes))
    hashSol.append(copy.deepcopy(routes))
    
    
    # Starting temperature
    T = obj*w/np.log(2)
    
              
    for itr in range(1, maxIter+1):
        if printout:
            print(" ")
            print("--------------------------------------------")
            print(" ")
            print("iteration", itr)
            print("T: ", T)
        
        # Copy current solution
        routes_copy = copy.deepcopy(routes)
        lR_copy = copy.deepcopy(lengthRoutes)
        rA_copy = copy.deepcopy(reqAssignment)
        
        # How many requests to remove
        q = np.random.randint(4, 1 + min(100, epsilon*instance["n"]))
        if printout:
            print("Nb. requests removed: ", q)
        
        # Select removal heuristic
        rem = np.random.choice([0,1,2], p = probaRem)
        nbUsedRem[rem] += 1
        if rem == 0:
            removedReq = remove.worst(instance, routes_copy, lR_copy, rA_copy, q, p_worst)
            if printout:
                print("Removal: worst")
        elif rem == 1:
            removedReq = remove.random(instance, routes_copy, lR_copy, rA_copy, q)
            if printout:
                print("Removal: random")
        else:
            removedReq = remove.shaw(instance, routes_copy, lR_copy, rA_copy, q, p, phi, psi)
            if printout:
                print("Removal: Shaw")
        
        
        # Insertion cost with or without noise
        withNoise = np.random.choice([False, True], p = probaNoise)
        if withNoise:
            nbUsedNoise[1] += 1
        else:
            nbUsedNoise[0] += 1
            
        # Select insertion heuristic
        ins = np.random.choice([0,1,2,3,4], p = probaIns)
        nbUsedIns[ins] += 1
        
        if ins == 0:
            insert.greedy(instance, routes_copy, lR_copy, removedReq, rA_copy, withNoise, eta)
            if printout:
                print("Insertion: Greedy, with noise: ", withNoise)
        elif rem == 1:
            insert.regret(instance, routes_copy, lR_copy, removedReq, 2, rA_copy, withNoise, eta)
            if printout:
                print("Insertion: Regret-2, with noise: ", withNoise)
        elif rem == 2:
            insert.regret(instance, routes_copy, lR_copy, removedReq, 3, rA_copy, withNoise, eta)
            if printout:
                print("Insertion: Regret-3, with noise: ", withNoise)
        elif rem == 3:
            insert.regret(instance, routes_copy, lR_copy, removedReq, 4, rA_copy, withNoise, eta)
            if printout:
                print("Insertion: Regret-4, with noise: ", withNoise)
        else:
            insert.regret(instance, routes_copy, lR_copy, removedReq, len(routes_copy), rA_copy, withNoise, eta)
            if printout:
                print("Insertion: Regret-m, with noise: ", withNoise)
    
        new_obj = sum(lR_copy)
        if printout:
            print("New Obj: ", new_obj)
        
        # Accepting criterion (Simulated Annealing)
        proba_accept = np.exp(-1*(new_obj - obj)/T)
        if new_obj <= obj or np.random.choice([True, False], p=(proba_accept, 1-proba_accept)):
            
            status = -1
            
            if new_obj < bestObj:
                bestObj = new_obj
                bestRoutes = copy.deepcopy(routes_copy)
                status = 0 #--> new best solution found
                obj_time.append((bestObj, time.time()-starttime))
                if printout:
                    print("Status: best solution found")
                if lightprintout:
                    print("---------------------")
                    print("Iteration:", itr)
                    print("New best solution found:", bestObj)
            else:
                # Check if new solution has been already accepted previously
                if not _wasAccepted(hashObj, hashLR, hashSol, new_obj, lR_copy, routes_copy):
                    if new_obj < obj :
                        status = 1 #--> new solution is better than the previous one
                        if printout:
                            print("Status: new solution found, better than current solution")
                    else :
                        status = 2 #--> new solution is worse than the previous one but still got accepted
                        if printout:
                            print("Status: new solution found, not better than current solution, accepted")
                      
                elif printout:
                    print("Status: already visited solution, accepted")
                    
                    
            # New solution is accepted so we update current solution    
            routes = routes_copy
            lengthRoutes = lR_copy
            reqAssignment = rA_copy
            obj = new_obj
                                                 
            if status >= 0:
                # Add new solution to hash tables (status >= 0 --> unvisited solution)
                hashObj.add(obj)
                idxInsert = hashObj.index(obj)
                hashLR.insert(idxInsert, copy.deepcopy(lR_copy))
                hashSol.insert(idxInsert, copy.deepcopy(routes_copy))
                
                # Heuristics score update 
                scoreIns[ins] += sigma[status]
                scoreRem[rem] += sigma[status]
                if withNoise:
                    scoreNoise[1] += sigma[status]
                else:
                    scoreNoise[0] += sigma[status]
                    
        else:
            if printout:
                print("Status: not accepted")
        
        
        # Update temperature
        T = c*T
        
        
        # Update heuristics weight at the end of every segment
        if itr % lengthSegment == 0:
            
            for k in range(3):
                if nbUsedRem[k] > 0:
                    weightRem[k] = weightRem[k]*(1-r) + r*scoreRem[k]/nbUsedRem[k]
                scoreRem[k] = 0
                nbUsedRem[k] = 0
            sumWeightRem = sum(weightRem)
            probaRem = [weightRem[k]/sumWeightRem for k in range(3)]
                        
            for k in range(5):
                if nbUsedIns[k] > 0:
                    weightIns[k] = weightIns[k]*(1-r) + r*scoreIns[k]/nbUsedIns[k]
                scoreIns[k] = 0
                nbUsedIns[k] = 0
            sumWeightIns = sum(weightIns)
            probaIns = [weightIns[k]/sumWeightIns for k in range(5)]
                
            for k in range(2):
                if nbUsedNoise[k] > 0:
                    weightNoise[k] = weightNoise[k]*(1-r) + r*scoreNoise[k]/nbUsedNoise[k]
                scoreNoise[k] = 0
                nbUsedNoise[k] = 0
            sumWeightNoise = sum(weightNoise)
            probaNoise = [weightNoise[k]/sumWeightNoise for k in range(2)]
                       
            if printout:
                print(" ")
                print("##################################")
                print("Computation of new heuristic weights")
                print("Weights Removal: worst: ", weightRem[0], ", random: ", weightRem[1], ", Shaw: ", weightRem[2] )
                print("Weights Insertion: greedy: ", weightIns[0], ", regret-2: ", weightIns[1], ", regret-3: ", weightIns[2], ", regret-4: ", weightIns[3],", regret-m: ", weightIns[4])
                print("Weight Noise: with ", weightNoise[0], ", without: ", weightNoise[1])
                print("##################################")
                print(" ")
        
    
    return bestObj, bestRoutes, obj_time
        