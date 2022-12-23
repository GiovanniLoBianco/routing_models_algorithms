#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os, sys


def _relatedness(instance, phi, psi):
    """
        Compute relatedness among every pair of requests.
        
        Relatedness is computed such as described in
        "An Adaptive Large Neighborhood Search Heuristics for the Pick-Up and Delivery Problems with Time Windows",
        S. Ropke and D. Pisinger, 2006.
        
        Default values for phi and psi are such as in the latter paper after parameter tuning.
        
        Store list of requests sorted by relatedness for each request. The ordering do not change during the solving process.
    """
    
    n = instance['n']
    t = instance['times']
    d = instance['demands']
    
    if "maxTimes" in instance.keys():
        maxTimes = instance["maxTimes"]
    else:
        maxTimes = np.max(instance["times"])
        instance["maxTimes"] = maxTimes
    
    maxDemands = np.max(d)
    
    r = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i+1,n):
            r[i][j] = phi*(t[i][j] + t[i+n][j+n])/maxTimes + psi*np.abs(d[i]-d[j])/maxDemands
            r[j][i] = r[i][j]
            
    instance["relatedness"] = {}
    for i in range(n):
        mostRelatedReq = [req for req in range(n) if req != i]
        mostRelatedReq.sort(key = lambda req: r[i][req])
        instance["relatedness"][i] = mostRelatedReq
    
    return r            


def shaw(instance, routes, lengthRoute, reqAssignment, q, p, phi = 9, psi = 2):
    
    """ 
        Shaw removal heuristic such as described in
        "An Adaptive Large Neighborhood Search Heuristics for the Pick-Up and Delivery Problems with Time Windows",
        S. Ropke and D. Pisinger, 2006.

        Remove a first request randomly, then remove request with high relatedness 
        regarding one of the previously removed requests (chosen randomly). Removing similar requests increase the chance
        to find better route configuration.
    
        - reqAssignement[req] gives the route that serve request req
        
        - q is number of requests to remove
        
        - p is a randomness parameter. The higher is p, the more related are the removed requests.
    """
    
    removedReq = []
    
    if q > 0:
    
        n = instance['n']
        if not "relatedness" in instance.keys():
            _relatedness(instance, phi, psi)

        req = np.random.randint(n) # Select first request randomly
        removedReq.append(req) # store every removed requests
        
        # Code optimization for access in constant time
        bitSet_removedReq = np.zeros(n)
        bitSet_removedReq[req] = 1
        bitSet_modifiedRoute = np.zeros(len(routes)) # Only modified routes will have their length updated
        bitSet_modifiedRoute[reqAssignment[req]] = 1

        # Remove req from current solution
        routes[reqAssignment[req]].remove(req)
        routes[reqAssignment[req]].remove(req + n)

        while len(removedReq) < q:

            req = np.random.choice(removedReq)

            # Compute rank of the related request of req to remove
            y = np.random.rand()
            rank = int(np.floor(np.power(y,p) * (n - len(removedReq))))

            # Get the request to remove
            idx = 0
            pos = -1
            while pos < rank:
                reqIdx = instance["relatedness"][req][idx]
                if bitSet_removedReq[reqIdx] == 0:
                    pos += 1
                idx += 1
            nextReq = instance["relatedness"][req][idx-1]

            # Remove nextReq from current solution
            removedReq.append(nextReq)
            bitSet_removedReq[nextReq] = 1

            routes[reqAssignment[nextReq]].remove(nextReq)
            routes[reqAssignment[nextReq]].remove(nextReq + n)

            bitSet_modifiedRoute[reqAssignment[nextReq]] = 1


        # Recompute length of routes
        for k in range(len(routes)):
            if bitSet_modifiedRoute[k] == 1:
                lengthRoute[k] = sum([instance["times"][routes[k][i]][routes[k][i+1]] for i in range(len(routes[k]) -1)])
    
    
    return removedReq


def random(instance, routes, lengthRoute, reqAssignment, q):
    
    """ 
        Random removal heuristic such as described in
        "An Adaptive Large Neighborhood Search Heuristics for the Pick-Up and Delivery Problems with Time Windows",
        S. Ropke and D. Pisinger, 2006.

        Remove q requests randomly.

        - reqAssignement[req] gives the route that serve request req

        - q is number of requests to remove

    """
    
    n = instance["n"]
    
    bitSet_modifiedRoute = np.zeros(len(routes))
    
    rndRequests = np.random.permutation(n)[0:q].tolist() # Generate uniformly randomly requests to remove
    
    # Remove requests from solution
    for req in rndRequests:
        routeOfReq = reqAssignment[req]
        bitSet_modifiedRoute[routeOfReq] = 1
        routes[routeOfReq].remove(req)
        routes[routeOfReq].remove(req + n)
    
    # Update length of routes
    for k in range(len(routes)):
        if bitSet_modifiedRoute[k] == 1:
            lengthRoute[k] = sum([instance["times"][routes[k][i]][routes[k][i+1]] for i in range(len(routes[k]) -1)])
    
    return rndRequests
    

def costRemoval(instance, route, pos_ori, pos_des):
    
    """
        Compute cost of removal, regarding distance, of the request with pick up location = pos_ori
        and delivery location = pos_des on the specified route.
          
    """
    
    t = instance["times"]
    n = instance["n"]
    
    assert pos_ori < pos_des, "destination must be after pickup"
    assert route[pos_ori] + n == route[pos_des], "pos_ori and pos_des must be position of origin and delivery of a same request"
    
    req = route[pos_ori]

    # Compute cost of removal for each position configuration
    cost = 0
    if pos_ori == pos_des-1:
        cost += t[req][req+n]
        if pos_ori > 0:
            cost += t[route[pos_ori-1]][req]
        if pos_des < len(route)-1:
            cost += t[req+n][route[pos_des+1]]
        if pos_ori > 0 and pos_des < len(route)-1:
            cost -= t[route[pos_ori-1]][route[pos_des+1]]           
    else:
        if pos_ori > 0:
            cost += t[route[pos_ori-1]][req] + t[req][route[pos_ori+1]] - t[route[pos_ori-1]][route[pos_ori+1]]
        else:
            cost += t[req][route[pos_ori+1]]

        if pos_des < len(route)-1:
            cost += t[route[pos_des-1]][req+n] + t[req+n][route[pos_des+1]] - t[route[pos_des-1]][route[pos_des+1]]
        else:
            cost += t[route[pos_des-1]][req+n]
    
    return cost
    

def worst(instance, routes, lengthRoute, reqAssignment, q, p):
    
    """ 
        Worst removal heuristic such as described in
        "An Adaptive Large Neighborhood Search Heuristics for the Pick-Up and Delivery Problems with Time Windows",
        S. Ropke and D. Pisinger, 2006.

        Remove request with high costs.
    
        - reqAssignement[req] gives the route that serve request req
        
        - q is number of requests to remove
        
        - p is a randomness parameter. The higher is p, the more costly are the removed requests.
    """
    
    n = instance["n"]
       
    # Code optimization: Computing position of origin and destination of each request --> access in constant time
    req_ori_des = [[-1,-1] for req in range(n)]
    for route in routes:
        for i in range(len(route)):
            if route[i] < n:
                req_ori_des[route[i]][0] = i
            else:
                req_ori_des[route[i]-n][1] = i
                
    # For each route, we compute the removal cost of each requests served on this route.
    # removalCosts[k] is the list of couple (cost, req) of removal cost for each request req served on route k
    removalCosts = [[] for k in range(len(routes))]
    for req in range(n):
        route_idx = reqAssignment[req]
        removalCosts[route_idx].append((costRemoval(instance, routes[route_idx], req_ori_des[req][0], req_ori_des[req][1]), req))
    
    removedReq = [] # List of removed requests
    
    for itr in range(q):
        
        # We sort all requests regarding their removal cost
        L = [costReq for k in range(len(routes)) for costReq in removalCosts[k]]
        L.sort(reverse=True)
        
        # Compute rank of the worst request to remove
        y = np.random.rand()
        rank = int(np.floor(np.power(y,p) * len(L)))
        (cost, req) = L[rank]
        
        # Remove req from solution
        route_idx = reqAssignment[req]
        del routes[route_idx][req_ori_des[req][1]]
        del routes[route_idx][req_ori_des[req][0]] 
        lengthRoute[route_idx] -= cost
        removedReq.append(req)
        
        # Update position of origin and destination
        for city in routes[route_idx]:
            if city < n:
                if req_ori_des[city][0] > req_ori_des[req][1]:
                    req_ori_des[city][0] -= 2
                    req_ori_des[city][1] -= 2
                else:
                    if req_ori_des[city][0] > req_ori_des[req][0]:
                        req_ori_des[city][0] -= 1
                    if req_ori_des[city][1] > req_ori_des[req][1]:
                        req_ori_des[city][1] -= 1
                    if req_ori_des[city][1] > req_ori_des[req][0]:
                        req_ori_des[city][1] -= 1
    
        # Update removal costs over the modified route
        removalCosts[route_idx] = []
        for city in routes[route_idx]:
            if city < n and not city in removedReq:
                removalCosts[route_idx].append((costRemoval(instance, routes[route_idx], req_ori_des[city][0], req_ori_des[city][1]), city))
    
      
    return removedReq
    
