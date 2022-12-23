#!/usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB


## Solve TSP with Gurobi
def solveWithGRB(n, times, timeLimit, accTimes=3, XP=False):
    
    try:
        # Create a new model
        model = gp. Model ("tsp")
        
        model.Params.LogToConsole = 0
        model.setParam('TimeLimit', timeLimit)
    
        # Create variables
        x = model.addVars(n,n, vtype=GRB.BINARY, name="x")
        u = model.addVars(n-1, vtype=GRB.INTEGER, name="u")
    
        # Set objective
        model.setObjective(gp.quicksum(x[i,j]*times[i][j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    
        # Onehot Constraint
        for j in range(n):
            model.addConstr(gp.quicksum(x[i,j] for i in range(n) if i!=j) == 1, name="onehot_col_"+str(j))
        for i in range(n):
            model.addConstr(gp.quicksum(x[i,j] for j in range(n) if i!=j) == 1, name="onehot_row_"+str(i))

        # Subtour Elimination Constraint
        for i in range(n-1):
            for j in range(n-1):
                if i!=j:
                    model.addConstr(u[i]-u[j]+(n-1)*x[i+1,j+1]<=n-2, name="subtours_"+str(i)+"_"+str(j))

        # Domain of u
        for i in range(n-1):
            model.addConstr(u[i]<=n-1, name="domain_u_1"+str(i))
            model.addConstr(1 <= u[i], name="domain_u_2"+str(i))
            
        # Callback function to get objective value every time a new solution is found (only when running XP)
        obj_time = []       
        def get_obj_time(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_time.append((model.cbGet(GRB.Callback.MIPSOL_OBJBST), model.cbGet(GRB.Callback.RUNTIME)))

        # Optimize
        if XP:
            model.optimize(get_obj_time)
        else:
            model.optimize()
        
        epsilon = 0.001
        if model.solCount > 0:
            route = [0]
            current = 0
            while len(route)<n:
                for i in range(1, n):
                    if i != current and x[current, i].X >=1 - epsilon:
                        route.append(i)
                        current = i
                        break
            return route, model.objVal, obj_time, model.Status == GRB.OPTIMAL
        else:
            return [], 0, obj_time, False
            
    except gp.GurobiError as e:
        print ("Error code" + str (e. errno ) + ":" + str(e))
    except AttributeError :
        print ("Encountered an attribute error ")





