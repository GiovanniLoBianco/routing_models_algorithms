#!/usr/bin/env python
# coding: utf-8


import os, sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve(instance, cost, timeLimit, printout=False):
    """
       Solve a quadratic assignment problem with capacity constraint.
       cost can be the distance or the angle gaps.
       
       Returns clusters minimizing the cost
    """
    
    n = instance["n"]-1
    m = instance["m"]
    demand = instance["demands"]
    C = instance["capacity"]
    
    try:
        # Create a new model
        model = gp. Model ("QAP")
        
        # Create variables
        x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    
        # Set objective
        model.setObjective(gp.quicksum(cost[i][j]*x[i,k]*x[j,k] for i in range(n) for j in range(n) for k in range(m)), GRB.MINIMIZE)
    
        # Onehot Constraint
        for i in range(n):
            model.addConstr(gp.quicksum(x[i,k] for k in range(m)) == 1, name="onehot_city_"+str(i))
        
        # Capacity constraint
        for k in range(m):
            model.addConstr(gp.quicksum(demand[i]*x[i,k] for i in range(n)) <= C, name="capacity_"+str(k))
     
        # Callback function to get objective value every time a new solution is found (only when running XP)
        obj_time = []       
        def get_obj_time(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_time.append((model.cbGet(GRB.Callback.MIPSOL_OBJBST), model.cbGet(GRB.Callback.RUNTIME)))

        # Optimize
        if not printout:
            model.Params.LogToConsole = 0
        model.setParam('TimeLimit', timeLimit)
        
        model.optimize()
        
        clusters = []
        for k in range(m):
            c = []
            for i in range(n):
                if x[i,k].X == 1:
                    c.append(i)
            clusters.append(c)
        
        return clusters, model.objVal
            
    except gp.GurobiError as e:
        print ("Error code" + str (e. errno ) + ":" + str(e))
        return [], 0
    except AttributeError :
        print ("Encountered an attribute error ")
        return [], 0



