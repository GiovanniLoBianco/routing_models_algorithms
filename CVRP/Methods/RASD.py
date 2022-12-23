#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time, datetime

import FlowModel
from ClusterAndTSP import ctsp, abctsp, tbctsp, atbctsp

# Return a first solution with angle-based clustering + TSP
def init_abctsp(instance):
    # Solving time : 100bits/s
    tlABC = int(np.ceil(instance['n']*instance['m']/300))
    tlTSP = 60
    
    return abctsp(instance, tlABC, tlTSP)



# Run Randomized Adaptive Spatial Decoupling algorithm on instance.

# initMode: function that takes only an instance as parameter 
# and returns a first solution: list of routes, length of routes, objective value

# useDA = True --> use DA to solve CVRP at each step. False --> use Gurobi

# sizeProblem(n, m, Vmax) is a method that return the size of the problem for the given parameters:
# number of variables for QUBO or linear models, for example

# maxSizeCVRP is the maximum number of variables in the model to solve CVRP at each step

# maxIter is the number of iteration rasd should run

# RASD consists of isolating one set of routes from the current solution 
# and reoptimizing it independently from the rest of the solution.
# The selection of the routes are as follows: 
# we first select a route randomly and select the following routes in anticlockwise order
# until the size of the induced subproblem exceed maxSizeCVRP.
# We then call DA or Gurobi to solve the CVRP and update the current solution, if a better solution has been found.
# We repeat until stop condition is met.
# Possible improvement --> accept worse solution such as in a Simulated Annealing framework.

# Returns solution: list of routes + objective, objective value of the best solution found at every iteration (array) + total waiting time on DA server.
def solve(instance, initMode, sizeProblem, maxSizeCVRP, maxIter, printout = False):
    
    if printout:
        print("Initialization...")
    
    ## First solution
    routes, obj, lengthRoutes = initMode(instance)
    
    obj_iter = [obj]
    obj_time = [(obj,0)]
    
    m = len(routes)
    
    if printout:
        print("First solution: "+str(obj))
        print(routes)
    
    # Sorting first routes by angle from -pi to pi
    meanAngle = []
    for route in routes:
        meanAngle.append(sum([instance['angles'][city-1] for city in route if city!=0])/(len(route)-1))        
    routesData = [[routes[k], lengthRoutes[k], meanAngle[k]] for k in range(m)]
    routesData.sort(key = lambda x: x[2])
    
    
    time_start = time.time()
    
    # Main loop
    for itr in range(maxIter):
        if printout:
            print("######## Iteration "+str(itr))
        
        # Initialize instance of subproblem to solve.
        # Is updated every time one route is added to the subproblem.
        # times and demands are added in the end as they do not have impact on the problem's size.
        subCVRP = {
            'n' : 0,
            'm' : 0,
            'Vmax' : instance['Vmax'],
            'capacity' : instance['capacity']
        }
        
        # List of cities included in the subproblem, initialized with depot
        cities = [0]
        
        # Indices of routes selected
        subRoutes = []
        
        # Map cities to their index in subproblem.
        # Index of depot in subproblem is 0
        mapCitiesIndex = dict()
        mapCitiesIndex[0] = 0
        
        # First route selected randomly
        k = np.random.randint(0, m)
        
        # Select nb of routes
        nbSubRoutes = np.random.randint(2, min(np.floor(m/2), 5)+1)
        
        subVmax = len(routesData[k][0])
        demandsinSubProblem = [instance["demands"][city-1] for city in routesData[k][0] if city!=0 ]
        
        # While next route fit the maximum subproblem's size, we insert it into subproblem
        while subCVRP['m'] < nbSubRoutes and sizeProblem(subCVRP['n'] + len(routesData[k][0]) - 1, subCVRP['m']+  1, subVmax) < maxSizeCVRP:
            
            subCVRP["Vmax"] = subVmax    
            
            # Update subproblem parameters
            subCVRP['n'] += len(routesData[k][0])-1
            subCVRP['m'] += 1
            
            # Add routes into subproblem
            subRoutes.append(k)
            
            # Update list of cities considered in subproblem and map them to their corresponding index
            for city in routesData[k][0]:
                if city !=0: # We do not re-add depot
                    cities.append(city)
                    mapCitiesIndex[city]=len(cities)-1
            
            # Select next route following anticlockwise order
            if k < m-1:
                k+=1
            else:
                k=0
            
            # Get next subVmax, based on demands in subproblem
            for city in routesData[k][0]:
                if city !=0:
                    demandsinSubProblem.append(instance["demands"][city-1])
            demandsinSubProblem.sort()
            
            subVmax = 0
            cumul = 0
            while demandsinSubProblem[subVmax] + cumul <= instance['capacity']:
                cumul += demandsinSubProblem[subVmax]
                subVmax += 1
            
        
        
        # Total length of routes considered in subproblem before reoptimizing them
        lengthRemovedRoutes = sum([routesData[r][1] for r in subRoutes])
        
        if printout:
            print("nb cities in sub CVRP: ", subCVRP['n'])
            print("nb routes in sub CVRP: ", subCVRP['m'])
            print("Vmax in sub CVRP: ", subCVRP['Vmax'])
            print("size sub CVRP: ", sizeProblem(subCVRP['n'], subCVRP['m'], subCVRP['Vmax']))
        
        # Complete subCVRP instances with traveled times and demands
        subCVRP['times'] = [[instance['times'][i][j] for j in cities] for i in cities]
        subCVRP['demands'] = [instance['demands'][i-1] for i in cities[1:]]
        #subCVRP['angleGaps'] = [[instance['angleGaps'][i-1][j-1] for j in cities[1:]] for i in cities[1:]]
        #subCVRP['coordinates'] = [instance['coordinates'][i] for i in cities]
        #subCVRP['angles'] = [instance['angles'][i-1] for i in cities[1:]]
        
        
        # Time limit for solving subproblem: --> 1s for 100 bits
        tlCVRP = int(np.ceil(sizeProblem(subCVRP['n'], subCVRP['m'], subCVRP['Vmax'])/300))
        
        # initialize solution of subproblems with current routes, formatted with subproblem indices
        initialRoutes = [[mapCitiesIndex[city] for city in routesData[r][0]] for r in subRoutes] 
        
        # Solve subCVRP
        newRoutes, newObj = FlowModel.solve(subCVRP, tlCVRP, ini = initialRoutes)
        
        
        # Update solution
        if len(newRoutes)>0 and newObj < lengthRemovedRoutes:
            obj += newObj-lengthRemovedRoutes
            
            obj_time.append((obj, time.time() - time_start - w_timeTotal))
        
            if printout:
                print("New solution: "+str(obj))
                
            # Convert routes into original cities index and compute length and mean angle
            newTrueRoutes = []
            newLengths = []
            newAngles = []
            for r in range(subCVRP['m']):
                newTrueRoute=[cities[idx] for idx in newRoutes[r]]
                newTrueRoutes.append(newTrueRoute)
                
                lengthNewRoute = sum([instance['times'][newTrueRoute[i]][newTrueRoute[i+1]] for i in range(len(newRoutes[r])-1)]) + instance['times'][newTrueRoute[-1]][newTrueRoute[0]]
                newLengths.append(lengthNewRoute)
                
                newAngle = sum([instance['angles'][city-1] for city in newTrueRoute if city != 0])/(len(newTrueRoute)-1)
                newAngles.append(newAngle)
        
            # Update set of routes and sort them
            for r in subRoutes:
                routesData[r][0] = newTrueRoutes.pop()
                routesData[r][1] = newLengths.pop()
                routesData[r][2] = newAngles.pop()
            routesData.sort(key = lambda x: x[2])
            
            if printout:
                rr = [rd[0] for rd in routesData]
                print(rr)
        
        
        
        # Add best objective found so far at this iteration
        obj_iter.append(obj)
        
    return routesData, obj, obj_iter, obj_time

def sizeProblem_DA(n, m, Vmax):
    return (n+1)*m*(Vmax+1)

def sizeProblem_GRB(n, m, Vmax):
    return (n+1)**2 + n

