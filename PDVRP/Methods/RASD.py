#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time, datetime

import MILPModel
import ClusterAndRouting as cr
import copy


## RASD

def solve(instance, initSol, sizeProblem, maxSizePDP, maxIter, p_select = 6, printout = False):
    
    """
        Run Randomized Adaptive Spatial Decoupling algorithm on instance.

        initSol: dictionary representing the initial solution (routes, objective, length of routes)

        useDA = True --> use DA to solve PDP at each step. False --> use Gurobi

        sizeProblem(n, m, Vmax) is a method that return the size of the problem for the given parameters:
        number of variables for QUBO or linear models, for example

        maxSizePDP is the maximum number of variables in the model to solve PDP at each step

        maxIter is the number of iteration rasd should run

        RASD consists in isolating one set of routes from the current solution 
        and reoptimizing it independently from the rest of the solution.
        The routes are chosen randomly uniformly at each step 
        Possible improvement --> accept worse solution such as in a Simulated Annealing framework.

        Returns solution: list of routes, length of routes, 
        objective, objective value of the best solution found at every iteration 
        and total waiting time on DA server.
    """
    
    
    ## First solution
    
    # Remove potential empty routes from initial solution
    routes = [route for route in initSol["routes"] if len(route)>0]
    obj = initSol["obj"]
    lengthRoute = [l for l in initSol["lengthRoute"] if l>0]
    
    # Max nb. of routes in subproblem
    maxRoutesSubProblem = 4
    
    # Start timer
    starttime = time.time()
    
    # Compute deviation among all requests and for all requests, compute the sorted list of its closest requests.
    deviation = cr.computeDeviation(instance)
    closestRequests = []
    for req in range(instance["n"]):
        closestRequests_req = [i for i in range(instance["n"]) if i != req]
        closestRequests_req.sort(key = lambda x: deviation[req][x])
        closestRequests.append(closestRequests_req)
        
    obj_iter = [obj] # Store best solution found at each iteration
    obj_time = [(obj,0)] # Store new best solution found and time
    
    if printout:
        print("First Solution: ", obj)
    
    # Main Loop   
    itr = 0
    while(itr < maxIter):
        if printout:
            print("######## Iteration "+str(itr))
        
        #### SELECT SUBPROBLEM
        
        # Select nb of routes
        nbSubRoutes = np.random.randint(2, min(np.floor(instance["m"]/2), maxRoutesSubProblem)+1)
        
        # Initialize instance of subproblem to solve.
        subPDP = {
            "name" : "subproblem",
            'n' : 0,
            'm' : 0,
            'Vmax' : instance['Vmax'],
            'capacity' : instance['capacity']
        }
        
        # If there are at least one unused vehicle, then we add it to the subproblem
        idleVehicle = 0
        if len(routes) < instance["m"]:
            idleVehicle+=1
        
        # Select a random route...
        first_route = np.random.randint(0, len(routes))
        subPDP["n"] += int(len(routes[first_route])/2)
        subPDP["m"] += 1
        removedRoutes = [first_route]
        
        cities = [] # Store every city in subproblem
        bitSet_requests = [0 for i in range(instance["n"])]
        for city in routes[first_route]:
            cities.append(city)
            if city < instance["n"]:
                bitSet_requests[city] = 1
        
        
        
        # Loop selecting a close (with a bit of randomness) request of current request that belongs to a different route
        # Add the associated route to the subproblem and select a new current route among all selected routes
        # Stop when we have reached nbSubRoutes or max size
        tooBig = False
        while subPDP["m"] < nbSubRoutes and not tooBig:
            
            # Select a random request among all already selected requests
            current_req = cities[np.random.randint(0, len(cities))]
            if current_req>=instance["n"]:
                current_req -= instance["n"]
            
            # selection of a close request
            y = np.random.random()
            idx_nextReq  = int( np.floor( np.power(y, p_select)*(instance["n"] - subPDP["n"]) ))
            count = -1
            for idx in range(instance["n"]):
                if bitSet_requests[closestRequests[current_req][idx]] == 0:
                    count+=1
                    if count == idx_nextReq:
                        next_req = closestRequests[current_req][idx]
                        break
            
            for k in range(len(routes)):
                if next_req in routes[k]:
                    new_route = k
                    break
            
            # check subproblem induced is not too large (must contain at least two routes)
            if subPDP["m"] < 2 or sizeProblem(subPDP['n'] + len(routes[new_route])/2, subPDP['m'] +  1 + idleVehicle, subPDP['Vmax']) <= maxSizePDP:
                
                # update subproblem
                removedRoutes.append(new_route)
                for city in routes[new_route]:
                    cities.append(city)
                    if city < instance["n"]:
                        bitSet_requests[city] = 1
                subPDP["m"] += 1
                subPDP["n"] += int(len(routes[new_route])/2)
            
            else:
                tooBig = True
        
        # Add idle vehicle to the problem
        subPDP["m"] += idleVehicle
        
        # Total length of routes considered in subproblem before reoptimizing them
        lengthRemovedRoutes = sum([lengthRoute[k] for k in removedRoutes])
      
        # Map cities to their index in subproblem.
        cities.sort() # We sort cities first to ensure request i has its pickup location at i and delivery location at i+n
        mapCitiesIndex = dict()
        i = 0
        for city in cities:
            mapCitiesIndex[city] = i
            i+=1
        
        # Select subarrays for traveling times and distance in subproblem
        subPDP['times'] = [[instance['times'][i][j] for j in cities] for i in cities]
        subPDP['demands'] = [instance['demands'][i] for i in cities]
        
        
        
        #### SOLVE SUBPROBLEM
        
        if printout:
            print("nb request in sub PDP: ", subPDP["n"])
            print("nb routes in sub PDP: ", subPDP["m"] - idleVehicle)
            print("nb variables in sub PDP: ", sizeProblem(subPDP["n"], subPDP["m"], subPDP["Vmax"]))
    
        # Time limit for solving subproblem: --> 1s for 50 bits
        tlPDP = int(np.ceil(sizeProblem(subPDP['n'], subPDP['m'], subPDP['Vmax'])/50))
        
        # Initial routes reindexed
        ini=[]
        for k in removedRoutes:
            ini.append([mapCitiesIndex[city] for city in routes[k]])
        
        # Solve subPDP
        newRoutes, newObj, _, _ = MILPModel.solve_RankBasedModel(subPDP, tlPDP, ini = ini)        

        
        #### UPDATE SOLUTION    
                 
        ## If DA found new routes and if these new routes are improving the current solution --> update solution
        if len(newRoutes) > 0 and newObj < lengthRemovedRoutes:
            
            removedRoutes.sort(reverse = True)
            
            # We remove previous routes
            for k in removedRoutes:
                del routes[k]
                del lengthRoute[k]
            
            # We add new routes
            for k in range(len(newRoutes)):
                routes.append([cities[idx] for idx in newRoutes[k]]) # re-index new routes before updating
                lengthRoute.append(sum([subPDP["times"][newRoutes[k][i]][newRoutes[k][i+1]] for i in range(len(newRoutes[k]) -1)]))
            
            # Update objective
            obj = obj - lengthRemovedRoutes + newObj    
            obj_time.append((obj, time.time()-starttime-w_timeTotal))
            
            if printout:     
                print("New Solution: ", obj)
        
        obj_iter.append(obj)
        
        itr+=1
    
    
    return routes, lengthRoute, obj, obj_iter, obj_time


def sizeProblem_DA(n, m, Vmax):
    return (2*n)*m*(Vmax)

def sizeProblem_GRB(n, m, Vmax):
    return (2*n+1)*(2*n+1)*m + 2*n + 2*n+1
    
    
    
    
    
    
    
    
    