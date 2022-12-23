#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import sys, time, datetime

import copy
import QAP

import MILPModel
import InsertionHeuristics as insert
from RemovalHeuristics import costRemoval


def computeDeviation(instance):
    """
        Compute deviation cost for every pair of requests
    """
    
    n = instance["n"]
    t = instance["times"]
    
    cost = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1, n):
            config_1 = t[i][i+n] + t[i+n][j] + t[j][j+n]
            config_2 = t[i][j] + t[j][i+n] + t[i+n][j+n]
            config_3 = t[i][j] + t[j][j+n] + t[j+n][i+n]
            config_4 = t[j][i] + t[i][i+n] + t[i+n][j+n]
            config_5 = t[j][i] + t[i][j+n] + t[j+n][i+n]
            config_6 = t[j][j+n] + t[j+n][i] + t[i][i+n]
            shortest = min([config_1, config_2,config_3,config_4,config_5,config_6])
            cost[i][j] = shortest - t[i][i+n] - t[j][j+n]
            cost[j][i] = cost[i][j]
    
    return cost




def clusterAndRouting(instance, tl_DBC, tl_R, printout):
    
    """
        First run deviation based clustering on the instance and then uncapacitated routing subproblem on each clusters
        Returns list of routes, objective value of the solution, length of each route.
        
        - tl_DBC (resp. tl_R) is the time limit to solve deviation based clustering (resp. each routing subproblem)
        
    """
    
    dev = computeDeviation(instance)

    # Get clusters
    clusters = QAP.solve(instance, dev, tl_DBC, printout)

    routes = []
    obj_total = 0
    length_route =[]
    
    nosolution = False # True iff no feasible solution was found for some reason
    
    for k, cluster in enumerate(clusters):

        if printout:
            print("--------------")
            print("Route "+str(k)+"...")

        
        # Generate routing problem on one cluster
        subTimes = [[instance["times"][i][j] for j in cluster] for i in cluster]
        subDemands = [instance["times"][i] for i in cluster]
        
        subRouting = {
            "n" : int(len(cluster)/2),
            "m" : 1,
            "Vmax" : len(cluster),
            "times" : subTimes,
            "demands" : subDemands,
            "capacity" : instance["capacity"]
        }
        
        # Solve routing subproblem (without capacity constraint)
        # We actually use the same model as for the PDP but with one vehicle.
        route, obj, _, _ = MILP.solve_RankBasedModel(subRouting, tl_R, relaxCapacity = True, printout = printout)
        if route == []:
            nosolution = True
            break
    
        # Update objective function and reindex cities of subproblem regarding the whole set of cities
        length_route.append(obj)
        obj_total += obj
        realRoute = [cluster[i] for i in route[0]]
        routes.append(realRoute)
        
    return routes, obj_total, length_route, nosolution


def removeWorstRequests(instance, routes, lengthRoute):
    
    """
        Remove requests in routes for which the capacity constraint is violated.
        While traveling along a route, whenever load exceeds capacity, we remove one by one requests currently carried
        until load fits capacity. The removed requests are the worst requests, regarding the gain in the objective value
        of their removal.
        
        Return list or removed requests and modify accordingly the routes in entry, 
        the objective value and the length of each route.
        
    """
    
    n = instance["n"]
    t = instance["times"]
    d = instance["demands"]
    C = instance["capacity"]
                 
    removedRequests = []
    
    # We check if each route violate capacity constraint and if so, we remove requests until capacity constraint is checked
    for k, route in enumerate(routes):
        
        # We travel along the route and update state of load and request aboard until load exceed capacity
        pos = 0
        while pos < len(route):
            aboard = np.zeros(n) # binary array, aboard[i] = 1 iff request i is in the vehicle at position pos
            load = 0
            while pos < len(route) and load <= C:
                if route[pos] < n: # if route[pos] is a pickup location...
                    aboard[route[pos]] = 1
                    load += d[route[pos]]
                else: #... or a delivery location
                    aboard[route[pos]-n] = 0
                    load += d[route[pos]]
                pos += 1
        
            # If current route violates capacity constraint...
            if pos < len(route): 
                potentialRemoval = [i for i in range(n) if aboard[i] == 1]
                
                #... then we remove some requests among requests currently carried by the vehicle at position pos
                # until load does not exceed capacity anymore.
                while load > C:   
                    # We select the worst request, i.e. the request, such that its removal is the largest gain in distance.
                    worstRequest = -1
                    bestGain = 0
                    for req in potentialRemoval: 
                        req_ori_pos = route.index(req)
                        req_des_pos = route.index(req+n)
                        
                        gain = costRemoval(instance, route, req_ori_pos, req_des_pos)
                        
                        # Update worst request 
                        if gain >= bestGain:
                            bestGain = gain
                            worstRequest = req
                    
                    # Remove worst request from route and update objective
                    route.remove(worstRequest)
                    route.remove(worstRequest + n)
                    lengthRoute[k] -= bestGain
                    
                    # Update current load
                    aboard[worstRequest] = 0
                    load -= d[worstRequest]
                    potentialRemoval.remove(worstRequest)
                    
                    # We add the removed request into the list of removed request
                    removedRequests.append(worstRequest)
          
    return removedRequests
                 

def solve(instance, tl_DBC, tl_R, printout = False):
                 
    """
        Solve the instance by first using deviation based clustering with DA 
        and then routing on each cluster (with relaxed capacity cosntraint).
        Then each route is fixed by removing the most costly unfit requests. 
        Removed requests are reinserted them later greedily.
    """
    
    # Cluster and uncapacitated routing
    if printout:
        print("Clusters and First Routes")
    routes, _, lengthRoute, nosolution = clusterAndRouting(instance, tl_DBC, tl_R, printout)
    
    if nosolution:
        return [], 0, [], 0, 0, True
    
    # Remove worst unfit requests from route
    removedRequests = removeWorstRequests(instance, routes, lengthRoute)              
    if printout:
        print("Removed Unfit Requests: "+str(removedRequests))
    nb_removed_req = len(removedRequests)
    
    
    list_heuristics = ["Greedy"]
    if instance["m"] > 1:
        list_heuristics.append("Regret-2")
    if instance["m"] > 2:
        list_heuristics.append("Regret-3")
    if instance["m"] > 3:
        list_heuristics.append("Regret-4")
    if instance["m"] > 4:
        list_heuristics.append("Regret-m")
    
    
    # We try all heuristics for fixing the solution and pick the best result
    
    best_solution = 2*instance["n"]*np.max(instance["times"])
    best_method = ""
    best_routes = []
    best_lengthRoute = []
    
    for method in list_heuristics:
        
        # We first make a deep copy of the current routes and requests to insert
        routes_copy = copy.deepcopy(routes)
        lengthRoute_copy = copy.deepcopy(lengthRoute)
        removedRequests_copy = copy.deepcopy(removedRequests)
        
        if method == "Greedy":
            insert.greedy(instance, routes_copy, lengthRoute_copy, removedRequests_copy)
        if method == "Regret-2":
            insert.regret(instance, routes_copy, lengthRoute_copy, removedRequests_copy, 2)
        if method == "Regret-3":
            insert.regret(instance, routes_copy, lengthRoute_copy, removedRequests_copy, 3)
        if method == "Regret-4":
            insert.regret(instance, routes_copy, lengthRoute_copy, removedRequests_copy, 4)
        if method == "Regret-m":
            insert.regret(instance, routes_copy, lengthRoute_copy, removedRequests_copy, instance["m"])
        
        obj_method = sum(lengthRoute_copy)
        if obj_method < best_solution:
            best_method = method
            best_solution = obj_method
            best_routes = routes_copy
            best_lengthRoute = lengthRoute_copy
        
    if printout:
        print("Objective: ", obj)
        print("Fixed with: ", best_method)
                                 
    return best_routes, best_solution, best_lengthRoute, nb_removed_req, nosolution



