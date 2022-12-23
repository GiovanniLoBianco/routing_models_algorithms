#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import time, datetime

import gurobipy as gp
from gurobipy import GRB


def solve(instance, timeLimit, relaxCapacity=False, printout = False, ini = [], XP = False):
    """
        Create a edge rank based linear model for the PDP on the given instance and solve it with Gurobi
        
        - relaxCapacity = True iff we do not consider capacity constraint in our problem.
            We relax capacity constraint within our clustering + routing approach.
        
        - ini is the initial state of routes, if given
        
    """
    
    n = instance["n"]
    m = instance["m"]
    Vmax = instance["Vmax"]
    times = instance["times"]
    demands = instance["demands"]
    C = instance["capacity"]
    
    # We add a dummy node for modeling purpose, that is at distance 0 from every node, indexed at 0.
    # And we set the travel times according to the given time accuracy.
    times_wDummy = np.zeros((2*n+1, 2*n+1), np.float16)
    for i in range(1, 2*n+1):
        for j in range(1, 2*n+1):
            times_wDummy[i][j] = times[i-1][j-1]
            
    try:
        
        # Create a new model
        model = gp.Model("PDP_Furtado_"+instance["name"])

        # Create variables
        x = model.addVars(2*n+1,2*n+1, vtype=GRB.BINARY, name="x")
        u = model.addVars(2*n, vtype=GRB.CONTINUOUS, name="u")
        q = model.addVars(2*n, vtype=GRB.CONTINUOUS, name="q")
        v = model.addVars(2*n, vtype=GRB.CONTINUOUS, name="v")
        
        # Set objective
        model.setObjective(gp.quicksum(x[i,j]*times_wDummy[i][j] for i in range(2*n+1) for j in range(2*n+1)), GRB.MINIMIZE)
    
        # Onehot Constraints
        for i in range(1, 2*n+1):
            model.addConstr(gp.quicksum(x[i,j] for j in range(2*n+1) if i!=j) == 1, name="onehot_city_1_"+str(i))
        for j in range(1, 2*n+1):
            model.addConstr(gp.quicksum(x[i,j] for i in range(2*n+1) if i!=j) == 1, name="onehot_city_2_"+str(j))
            
        # Flow depot
        model.addConstr(gp.quicksum(x[0,j] for j in range(1,2*n+1)) <= m, name="nb_vehicles")
        model.addConstr(gp.quicksum(x[0,j] for j in range(1,2*n+1)) == gp.quicksum(x[i,0] for i in range(1,2*n+1)), name="flow_depot")
        
        # Subtour Elimination MZT
        for i in range(1,2*n+1):
            for j in range(1, 2*n+1):
                if i!=j:
                    model.addConstr(u[j-1] >= u[i-1] + 1 - Vmax*(1 - x[i,j]), name = "subtour_"+str(i)+"_"+str(j))
        for i in range(2*n):
            model.addConstr(u[i] >= 1, name="lb_u_"+str(i))
            model.addConstr(u[i] <= Vmax, name="ub_u_"+str(i))
        
        for i in range(n):
            model.addConstr(u[i] + 1 <= u[n+i], name="pickup_before_delivery_"+str(i))
            
        # Load changes
        for i in range(1,2*n+1):
            for j in range(1, 2*n+1):
                if i!=j:
                    model.addConstr(q[j-1] >= q[i-1] + demands[i-1] - (C + demands[i-1])*(1 - x[i,j]), name = "load_"+str(i)+"_"+str(j))
        
        # Capacity constraint
        for i in range(n):
            model.addConstr(demands[i] <= q[i], name = "lb_pickup_q_"+str(i))
            model.addConstr(q[i] <= C, name = "ub_pickup_q_"+str(i))
        for i in range(n, 2*n):
            model.addConstr(0 <= q[i], name = "lb_delivery_q_"+str(i))
            model.addConstr(q[i] <= C - demands[i], name = "ub_delivery_q_"+str(i))
            
        # Index route consistency
        for i in range(n):
            model.addConstr(v[i+n] == v[i], name = "pickup_delivery_same_route_"+str(i))
        for i in range(2*n):
            model.addConstr(v[i] >= (i+1)*x[0,i+1], name = "first_visit_index_1_"+str(i))
            model.addConstr(v[i] <= (i+1)*x[0,i+1] + n*(1 - x[0,i+1]), name = "first_visit_index_2_"+str(i))
        for i in range(1,2*n+1):
            for j in range(1, 2*n+1):
                model.addConstr(v[j-1] >= v[i-1] - n*(1 - x[i,j]), name = "same_route_same_index_1_"+str(i)+"_"+str(j))
                model.addConstr(v[j-1] <= v[i-1] + n*(1 - x[i,j]), name = "same_route_same_index_2_"+str(i)+"_"+str(j))
        

        ## Initial routes
        if ini != []:
            for i in range(1,2*n+1):
                for j in range(1, 2*n+1):
                    x[i, j].Start = 0
            for k, route in enumerate(ini):
                for pos in range(len(route)-1):
                    x[route[pos]+1, route[pos+1]+1].Start = 1
                x[0, route[0]+1].Start = 1
                x[route[-1]+1, 0].Start = 1
                    
        # Callback function to get objective value every time a new solution is found (only when running XP)
        obj_time = []       
        def get_obj_time(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_time.append((model.cbGet(GRB.Callback.MIPSOL_OBJBST), model.cbGet(GRB.Callback.RUNTIME)))
                                   
        # Solve
        if not printout:
            model.Params.LogToConsole = 0
        model.setParam('TimeLimit', timeLimit)
        if XP:
            model.optimize(get_obj_time)
        else:
            model.optimize()
        
        
        epsilon = 0.001
        if model.solCount > 0:
            routes = []
            for j in range(1, 2*n+1):
                if x[0,j].X >= 1 - epsilon: 
                    route = []
                    current = j
                    while current != 0:
                        route.append(current-1)
                        for jj in range(2*n+1):
                            if x[current, jj].X >= 1 - epsilon:
                                current = jj
                                break
                    routes.append(route)
                if len(routes) == m:
                    break
            return routes, model.objVal, obj_time, model.Status == GRB.OPTIMAL                
        else:
            return [], 0, obj_time, False

    except gp.GurobiError as e:
        print ("Error code" + str (e. errno ) + ":" + str(e))
    except AttributeError as e:
        print ("Encountered an attribute error ")
        print (e)    
    