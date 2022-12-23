#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys


def _costInsertion(instance, route, req, pos1, pos2, withNoise, eta):
    
    """
        Return the cost of inserting request req in route at position pos1 (pickup) and pos2 (delivery)
        The ordering of other visits remains unchanged.
        
        In the ALNS, we can add noise in the insertion cost to enhance exploration. 
        Eta defines the intensity of the noise compared to the largest traveling time.
        
    """
    
    t = instance["times"]
    n = instance["n"]
    
    # If route has alreavy Vmax visits, then the cost is set large enough, so that it is not chosen
    if len(route) >= instance["Vmax"]:       
        if "maxTimes" in instance.keys():
            maxTimes = instance["maxTimes"]
        else:
            maxTimes = np.max(instance["times"])
            instance["maxTimes"] = maxTimes        
        return 4*maxTimes
    
    
    # Compute the cost of inserting req (resp. req+n) at position  pos1 (resp. pos2)
    cost = 0
    if pos1 == pos2-1:
        cost += t[req][req+n]
        if pos1 > 0 and pos2 < len(route)+1:
            cost -= t[route[pos1-1]][route[pos1]]
        if pos1 > 0:
            cost += t[route[pos1-1]][req]
        if pos2 < len(route) + 1:
            cost += t[req + n][route[pos1]] 
    else:
        cost += t[req][route[pos1]]
        cost += t[req+n][route[pos2-2]]
        if pos1 > 0:
            cost += t[req][route[pos1-1]]-t[route[pos1-1]][route[pos1]]
        if pos2 < len(route)+1:
            cost += t[req+n][route[pos2-1]]-t[route[pos2-2]][route[pos2-1]]
    
    # Adding noise
    if withNoise:
        if "maxTimes" in instance.keys():
            maxTimes = instance["maxTimes"]
        else:
            maxTimes = np.max(instance["times"])
            instance["maxTimes"] = maxTimes 
        maxN = eta * maxTimes
        noise = np.random.randint(-maxN, maxN+1)
        return max(0, cost+noise)
    else:
        return cost



def _getLoad(instance, route):
    """
        Return load of the vehicle at each position (at arrival) on the route.
    """
    d = instance["demands"]

    load_route = []
    for pos, city in enumerate(route):
        if pos == 0:
            load_route.append(0)
        else:
            load_route.append(load_route[pos-1]+d[route[pos-1]])

    return load_route



def _getPossiblePosition(instance, route, load, requests):
    
    """
        Return list of possible positions (pos1, pos2) for the insertion of each request on given route, 
        regarding the capacity constraint.
        Each request with same demand should have the same list of possible insertion, 
        so instead of computing possible positions for each request independently, 
        we reason on the quantity to be transported and the load of the vehicle at each position.
        We take advantage of the fact that a request with demand d can be inserted in at least 
        all possible positions of a request with demand d+l (l>0).
        
    """
    
    d = instance["demands"]
    C = instance["capacity"]
    
    # Get each demands' quantity (no duplicate).
    quantity = list(set([d[req] for req in requests]))
    
    # in descending order, because we start the computation of unfit positions from the largest requests
    # as it can be used to compute unfit positions for smaller requests.
    quantity.sort(reverse = True)
    
    # Store unfit positions for each demands' quantity, 
    # i.e. unfitPos[q] is the list of positions on the route for which 
    # the vehicle cannot carry an additional request with demand q.
    unfitPos = dict()

    # list of current unfit position, initialized with list of every position on the route
    currentUnfit = [pos for pos in range(len(route))]
    
    for q in quantity:
        
        # Unfit positions for a quantity q is based on unfit positions for quantity q+1 (or q+l)
        # We do not need to check if capacity constraint is checked on position that were
        # alredy checked with larger quantity.
        currentUnfit = [pos for pos in currentUnfit if load[pos] + q > C]
        unfitPos[q] = currentUnfit.copy()
    
    # Store possible positions (pos1, pos2) for insertion for each demands' quantity, 
    # i.e. (pos1, pos2) is in possPos[q] iff requests with demand q can be carried
    # on the route from pos1 to pos2. These positions refer to positions in the route after insertion,
    # not the original one, hence pos1 is in {0, ..., len(route)} and pos2 in {1, ..., len(route)+1}.
    possPos = dict()
    
    # The possible positions (pos1, pos2) are such that there is no unfit position between pos1 and pos2.
    # Hence, we identify each "blank" sections defined by empty space between each entry in unfitPos[q].
    # If unfitPos[q] = {...,a,b,...} then request with demand q can be inserted in the route 
    # such that it is picked up in position pos1 in {a+1,...,b-1} and drop off at position pos2 in {pos1+1,...,b}.
    for q in quantity:
        possPos[q] = []
        
        # We extend unfitPos[q] to consider possible insertion at the beginning and at the end of the route.
        ext_unfitPos_q = [-1] + unfitPos[q] + [len(route)+1] 
        for idx in range(len(ext_unfitPos_q)-1):
            for pos1 in range(ext_unfitPos_q[idx]+1, ext_unfitPos_q[idx+1]):
                for pos2 in range(pos1+1, ext_unfitPos_q[idx+1]+1):
                    possPos[q].append((pos1, pos2))
    
    # For each request, we simply assign the list of possible positions corresponding to its quantity
    req_possPos = dict()
    for req in requests:
        req_possPos[req] = possPos[d[req]]
                
    return req_possPos



def _computeBestInsertion(instance, routes, requests, withNoise, eta):
    
    """
        Return a dictionary bestInsertion. bestInsertion[(k, req)] = (cost, pos1, pos2) 
        where pos1 and pos2 are the best positions for inserting req on route k 
        and cost is the associated change in the objective value.
    """
    
    if "maxTimes" in instance.keys():
        maxTimes = instance["maxTimes"]
    else:
        maxTimes = np.max(instance["times"])
        instance["maxTimes"] = maxTimes
        
    bestInsertion = dict()
    
    for k, route in enumerate(routes):
        
        # Get possible positions for insertion regarding capacity constraint
        load = _getLoad(instance, route)
        possPos = _getPossiblePosition(instance, route, load, requests)
        
        # Select positions that minimize the cost in distance
        for req in requests:
            bestPos1 = 0
            bestPos2 = 1
            bestCost = 4*maxTimes
            for (pos1, pos2) in possPos[req]:
                cost = _costInsertion(instance, route, req, pos1, pos2, withNoise, eta)
                if bestCost > cost:
                    bestCost = cost
                    bestPos1 = pos1
                    bestPos2 = pos2
            bestInsertion[(k, req)] = (bestCost, bestPos1, bestPos2)
    
    return bestInsertion
    


def greedy(instance, routes, lengthRoute, requests, reqAssignment=[], withNoise = False, eta = 0.025):
    
    """
        Greedy insertion heuristic, such as described in 
        "An Adaptive Large Neighborhood Search Heuristics for the Pick-Up and Delivery Problems with Time Windows",
        S. Ropke and D. Pisinger, 2006.
        
        Select the best possible insertion at each step. Stop when all requests have been inserted.
        
        routes: current solution (list of routes).
        
        lengthRoute: list of length of each route.
        
        requests: requests to insert.
        
        reqAssignment: array such that reqAssignment[req] gives assigned route index. It is an optional argument,
        as it is used only when this method is called within the ALNS to gain some time in finding the route associated
        with each request. We do not need this feature when just fixing solution after clustering+routing for example.
        
    """
    
    n = instance["n"]
    if "maxTimes" in instance.keys():
        maxTimes = instance["maxTimes"]
    else:
        maxTimes = np.max(instance["times"])
        instance["maxTimes"] = maxTimes
    
    # We might want to consider unused vehicle when inserting requests.
    for k in range(len(routes), instance["m"]):
        routes.append([])
        lengthRoute.append(0)
    
    # Compute all best insertions for each pair (route, request)
    bestInsertion = _computeBestInsertion(instance, routes, requests, withNoise, eta)
    
    # We add one by one every request
    while len(requests) > 0:
    
        # We select the insertion with the least cost
        bestRoute = 0
        bestReq = 0
        bestCost = 4*maxTimes
        for k in range(len(routes)):
            for req in requests:
                cost, pos1, pos2 = bestInsertion[(k, req)]
                if bestCost > cost:
                    bestCost = cost
                    bestRoute = k
                    bestReq = req
        
        # We update routes, objective values, length of modified routes and list of remaining requests
        _, pos1, pos2 = bestInsertion[(bestRoute, bestReq)]
        
        # If we used noise we need to update route length with the actual insertion cost
        if withNoise:
            lengthRoute[bestRoute] +=  _costInsertion(instance, routes[bestRoute], bestReq, pos1, pos2, False, 0)
        else:
            lengthRoute[bestRoute] += bestCost
        
        routes[bestRoute].insert(pos1, bestReq)
        routes[bestRoute].insert(pos2, bestReq + n)
        requests.remove(bestReq)
        
        if len(reqAssignment) > 0:
            reqAssignment[bestReq] = bestRoute 
        
        # We recompute best insertion for remaining requests and modified route only
        # and update entries in bestInsertion
        new_bestInsertion = _computeBestInsertion(instance, [routes[bestRoute]], requests, withNoise, eta)
        for req in requests:
            bestInsertion[(bestRoute, req)] = new_bestInsertion[(0, req)]



def regret(instance, routes, lengthRoute, requests, K, reqAssignment=[], withNoise = False, eta = 0.025):
    
    """
        Regret insertion heuristic, such as described in 
        "An Adaptive Large Neighborhood Search Heuristics for the Pick-Up and Delivery Problems with Time Windows",
        S. Ropke and D. Pisinger, 2006.
        
        Select the insertion with the most regret at each step. Stop when all requests have been inserted.
        
        routes: current solution (list of routes).
        
        lengthRoute: list of length of each route.
        
        requests: requests to insert.
        
        reqAssignment: array such that reqAssignment[req] gives assigned route index. It is an optional argument,
        as it is used only when this method is called within the ALNS to gain some time in finding the route associated
        with each request. We do not need this feature when just fixing solution after clustering+routing for example.
        
        K: depth of regret considered. K=1 is equivalent to greedy insertion.
                
    """
    
    n = instance["n"]
    if "maxTimes" in instance.keys():
        maxTimes = instance["maxTimes"]
    else:
        maxTimes = np.max(instance["times"])
        instance["maxTimes"] = maxTimes
    
    # We might want to consider unused vehicle when inserting requests.
    for k in range(len(routes), instance["m"]):
        routes.append([])
        lengthRoute.append(0)
    
    # Compute all best insertions for each pair (route, request)
    bestInsertion = _computeBestInsertion(instance, routes, requests, withNoise, eta)
    
    # We add one by one every request
    while len(requests) > 0:
        
        # We select the insertion with largest regret
        bestReq = 0
        bestRegret = 0
        bestCost = 4 * maxTimes
        bestRoute = 0
        
        for req in requests:
            
            # We sort best insertion associated with one request.
            # We have a list of ((cost, pos1, pos2) , k) sorted in ascending order regarding the cost
            # where the best insertion of req is on route k on position pos1 and pos2.
            # This sorted costs allow the computation of the regret
            sorted_bestInsertion_req = [(bestInsertion[(k, req)], k) for k in range(len(routes))]
            sorted_bestInsertion_req.sort()
            
            # Compute regret
            regret = sum([sorted_bestInsertion_req[i][0][0] - sorted_bestInsertion_req[0][0][0] for i in range(K)])
            
            # Update the bestReq and bestRoute if regret is larger
            # or if regret is equal but the cost of insertion is smaller (tie breaking rule)
            if (regret == bestRegret and sorted_bestInsertion_req[0][0][0] < bestCost) or regret > bestRegret:
                bestRegret = regret
                bestReq = req
                bestCost = sorted_bestInsertion_req[0][0][0]
                bestRoute = sorted_bestInsertion_req[0][1]
              
        
        # We update routes, objective values, length of modified routes and list of remaining requests
        _, pos1, pos2 = bestInsertion[(bestRoute, bestReq)]
        
        # If we used noise we need to update route length with the actual insertion cost
        if withNoise:
            lengthRoute[bestRoute] +=  _costInsertion(instance, routes[bestRoute], bestReq, pos1, pos2, False, 0)
        else:
            lengthRoute[bestRoute] += bestCost  
        
        routes[bestRoute].insert(pos1, bestReq)
        routes[bestRoute].insert(pos2, bestReq + n)
        requests.remove(bestReq)
        
        if len(reqAssignment) > 0:
            reqAssignment[bestReq] = bestRoute
        
        # We recompute best insertion for remaining requests and modified route only
        # and update entries in bestInsertion
        new_bestInsertion = _computeBestInsertion(instance, [routes[bestRoute]], requests, withNoise, eta)
        for req in requests:
            bestInsertion[(bestRoute, req)] = new_bestInsertion[(0, req)]
    

