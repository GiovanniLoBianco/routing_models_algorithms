#!/usr/bin/env python
# coding: utf-8

import os, sys
import numpy as np
import json

import gurobipy as gp
from gurobipy import GRB


##-------------------------------------------------
##-------------- SOLVE WITH GUROBI ----------------
##-------------------------------------------------


def solve(instance, timeLimit, printout=False, ini=[], XP = False):
    
    """
    Solve linear model and return the solution + objective value if one has been found within the time limit
    printout = True iff stats should be printed during solving
    we can initialize the search at an initial solution stored in ini
    accTime is the accuracy for the times coefficient to consider, to compare with DA that only considers integer
    """
    
    n = instance['n']
    m = instance['m']
    Vmax = instance["Vmax"]
    C = instance['capacity']
    d = [0]+instance['demands']
    times = instance['times']

    try:
        # Create a new model
        model = gp. Model ("vrpCommodityFlow")

        # Create variables
        x = model.addVars(n+1,n+1, lb=0, ub=1, vtype=GRB.BINARY, name="x")
        r = model.addVars(n, vtype=GRB.INTEGER, name="r")
        
        # Set objective
        model.setObjective(gp.quicksum(x[i,j]*times[i][j] for i in range(n+1) for j in range(n+1)), GRB.MINIMIZE)
        
        # Onehot Constraint
        for j in range(1, n+1):
            model.addConstr(gp.quicksum(x[i,j] for i in range(n+1) if i!=j) == 1, name="onehot_col_"+str(j))
        for i in range(1, n+1):
            model.addConstr(gp.quicksum(x[i,j] for j in range(n+1) if i!=j) == 1, name="onehot_row_"+str(i))
            
        # Depot Flow Constraint
        model.addConstr(gp.quicksum(x[0,j] for j in range(1, n+1)) == gp.quicksum(x[i,0] for i in range(1, n+1)), name="depot_flow")
        model.addConstr(gp.quicksum(x[i,0] for i in range(1, n+1)) <= m, name="maxVehicles")
        for i in range(0, n+1):
            model.addConstr(x[i,i]==0, name="havetomove_"+str(i))
            
        # Capacity Constraint
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i!=j:
                    model.addConstr(-r[j-1]+r[i-1] - d[i]*x[i,j]+C*(1-x[i,j]) >= 0, name="capa_"+str(i)+"_"+str(j))
        
        # Lower and upper bounds of r
        for i in range(n):
            model.addConstr(r[i] >= d[i+1] , name = "lb_r_"+str(i))
            model.addConstr(r[i] <= C , name = "ub_r_"+str(i))
        
        
        # Add initial solution
        for k in range(len(ini)):
            for i in range(len(ini[k])-1):
                x[ini[k][i], ini[k][i+1]].Start = 1
         
        
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
        
        # Compute list of routes from solution
        epsilon = 0.001
        if model.solCount > 0:
            routes = []
            for j in range(1, n+1):
                # Gurobi does not convert binary value to their actual integer values and let precision error in the solution
                # which can make you lose days of your time when running XP...
                if x[0,j].X >= 1 - epsilon: 
                    route = [0]
                    current = j
                    while current != 0:
                        route.append(current)
                        for jj in range(n+1):
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
    except AttributeError :
        print ("Encountered an attribute error ")
      
    