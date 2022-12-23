#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import time, datetime

import gurobipy as gp
from gurobipy import GRB


def solve_RankBasedModel(instance, timeLimit, relaxCapacity=False, printout = False, ini = [], XP = False):
    """
        Create an edge rank based linear model for the PDP on the given instance and solve it with Gurobi
        
        - relaxCapacity = True iff we do not consider capacity constraint in our problem.
            We relax capacity constraint within our clustering + routing approach.
        
        - ini is the initial state of routes if given
        
        - XP is True if we record the evolution of incumbent over time
        
    """
          
    n = instance["n"]
    m = instance["m"]
    Vmax = instance["Vmax"]
    times = instance["times"]
    demands = instance["demands"]
    C = instance["capacity"]
          
          
    # We add a dummy node for modeling purpose, that is at distance 0 from every node, indexed at 2*n.
    # And we set the travel times according to the given time accuracy.
    times_wDummy = np.zeros((2*n+1, 2*n+1), np.float16)
    for i in range(2*n-1):
        for j in range(i+1, 2*n):
            times_wDummy[i][j] = times[i][j]
            times_wDummy[j][i] = times_wDummy[i][j]
                

    try:
        # Create a new model
        model = gp.Model("PDP_RankBased_"+instance["name"])

        # Create variables
        x = model.addVars(2*n+1,2*n+1, Vmax-1, m, vtype=GRB.BINARY, name="x")
        y = model.addVars(2*n+1, Vmax, m, vtype=GRB.BINARY, name="y")
        
        # Set objective
        model.setObjective(gp.quicksum(x[i,j,p,k]*times_wDummy[i][j] for i in range(2*n+1) for j in range(2*n+1) for k in range(m) for p in range(Vmax-1)), GRB.MINIMIZE)
        
        # Onehot Constraint
        for p in range(Vmax):
            for k in range(m):
                model.addConstr(gp.quicksum(y[i,p,k] for i in range (2*n+1)) == 1, name="onehot_pos_"+str(p)+"_"+str(k))
        for i in range(2*n):
            model.addConstr(gp.quicksum(y[i,p,k] for p in range(Vmax) for k in range(m)) == 1, name="onehot_city_"+str(i))
        
        if not relaxCapacity:
            # Capacity Constraint
            for k in range(m):
                for q in range(Vmax):
                    model.addConstr(gp.quicksum(demands[i]*y[i,p,k] for p in range(q+1) for i in range(2*n)) <= C, name="capacity"+str(k)+"_"+str(q))
                           
        # Pick Up before Delivery
        for i in range(n):
            model.addConstr(gp.quicksum((p+1)*(y[i,p,k]-y[i+n,p,k]) for p in range(Vmax) for k in range(m))+1<=0, name="order_"+str(i))

        # Pick Up and Delivery on Same Route
        for i in range(n):
            for k in range(m):
                model.addConstr(gp.quicksum(y[i+n,p,k] - y[i,p,k] for p in range(Vmax)) == 0, name="same_route_"+str(i)+"_"+str(k))
             
        # Dummy node is final
        for k in range(m):
            for p in range(Vmax-1):
                for q in range(p+1, Vmax):
                    model.addConstr(y[2*n,p,k] <= y[2*n,q,k], name="dummy_"+str(k)+"_"+str(p)+"_"+str(q))
                           
        # Binding x and y
        for k in range(m):
            for i in range(2*n+1):
                for p in range(Vmax-1):
                    model.addConstr(gp.quicksum(x[i,j,p,k] for j in range(2*n+1)) == y[i,p,k], name="bind_out_"+str(k)+"_"+str(i)+"_"+str(p))
                for p in range(1, Vmax):
                    model.addConstr(gp.quicksum(x[j,i,p-1,k] for j in range(2*n+1)) == y[i,p,k], name="bind_in_"+str(k)+"_"+str(i)+"_"+str(p))
        
                 
        ## Initial routes
        if ini != []:
            for k, route in enumerate(ini):
                for pos in range(length(route)-1):
                    x[route[pos], route[pos+1], pos, k].Start = 1
                    y[route[pos], pos, k].Start = 1
                y[route[length(route)-1], length(route)-1, k].Start = 1
                
                
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
                         
                         
        # Return list of routes from solution and objective value
        if model.solCount > 0:
            routes = []
            for k in range(m):
                route = []
                for pos in range(Vmax):
                    for city in range(2*n):
                        if y[city, pos, k].X == 1:
                            route.append(city)
                routes.append(route)
            return routes, model.objVal, obj_time, model.Status == GRB.OPTIMAL                 
        else:
            return [], 0, obj_time, False # No feasible solution found              
        

    except gp.GurobiError as e:
        print ("Error code" + str (e. errno ) + ":" + str(e))
    except AttributeError as e:
        print ("Encountered an attribute error ")
        print (e)



def solve_FlowModel(instance, timeLimit, relaxCapacity=False, printout = False, ini = []):
    
    """
       Create a flow linear model for the PDP on the given instance and solve it with Gurobi
       
        - relaxCapacity = True iff we do not consider capacity constraint in our problem.
            We relax capacity constraint within our clustering + routing approach.
        
        - ini is the initial state of routes if given
        
        - XP is True if we record the evolution of incumbent over time

    """
          
    n = instance["n"]
    m = instance["m"]
    times = instance["times"]
    demands = instance["demands"]
    Vmax = instance["Vmax"]
    C = instance["capacity"]
          
          
    # We add a dummy node for modeling purpose, that is at distance 0 from every node, indexed at 2*n.
    # And we set the travel times according to the given time accuracy.
    times_wDummy = np.zeros((2*n+1, 2*n+1), np.float16)
    for i in range(2*n-1):
        for j in range(i+1, 2*n):
            times_wDummy[i][j] = times[i][j]
            times_wDummy[j][i] = times_wDummy[i][j]
     
    try:
        # Create a new model
        model = gp. Model ("pickup_and_delivery_three_index")


        # Create variables
        x = model.addVars(2*n+1, 2*n+1, m, vtype=GRB.BINARY, name="x")
        u = model.addVars(2*n, vtype=GRB.INTEGER, name="u")
        r = model.addVars(2*n+1, vtype=GRB.INTEGER, name="r")
        
        # Set objective
        model.setObjective(gp.quicksum(x[i,j,k]*times_wDummy[i][j] for i in range(2*n+1) for j in range(2*n+1) for k in range(m)), GRB.MINIMIZE)
        
        # Onehot Constraint
        for i in range(2*n):
            model.addConstr(gp.quicksum(x[i,j,k] for j in range(2*n+1) for k in range(m) if i!=j) == 1, name="onehot_city_depart"+str(i))
        for j in range(2*n):
            model.addConstr(gp.quicksum(x[i,j,k] for i in range(2*n+1) for k in range(m) if i!=j) == 1, name="onehot_city_arrival"+str(j))
        
        # Flow for each vehicle
        for k in range(m):
            for i in range(2*n):
                model.addConstr(gp.quicksum(x[i,j,k] for j in range(2*n+1)) == gp.quicksum(x[j,i,k] for j in range(2*n+1)), name="flow_"+str(k)+"_"+str(i))
        
        # Flow Dummy node
        for k in range(m):
            model.addConstr(gp.quicksum(x[2*n,j,k] for j in range(2*n+1)) == 1, name="depart_dummy_"+str(k))
            model.addConstr(gp.quicksum(x[i,2*n,k] for i in range(2*n+1)) == 1, name="arrival_dummy_"+str(k))
        
        
        M = 3*C*n # Big-M
        
        # Precedence Constraint
        for k in range(m):    
            for i in range(2*n):
                for j in range(2*n):
                    model.addConstr( u[j] >= u[i]+1 - M*(1-x[i,j,k]), name="sequence_"+str(k)+"_"+str(i)+"_"+str(j))
        for i in range(n):
            model.addConstr(u[i+n] >= u[i] +1, name="pickup_before_delivery_"+str(i))
        
        if not relaxCapacity:
            # Capacity Constraint
            for k in range(m):    
                for i in range(2*n+1):
                    for j in range(2*n):
                        model.addConstr( r[j] >= r[i] + demands[j] - M*(1-x[i,j,k]), name="capacity_"+str(k)+"_"+str(i)+"_"+str(j))
                            
        # Pick-Up and Delivery on the same route
        for i in range(n):
            for k in range(m):
                model.addConstr(gp.quicksum(x[i,j,k] for j in range(2*n+1)) == gp.quicksum(x[i+n,j,k] for j in range(2*n+1)), name="same_route_"+str(k)+"_"+str(i))
        
        # Domains of commodities and 
        for i in range(2*n):
            model.addConstr(r[i]>=0)
            model.addConstr(r[i]<=C)
            model.addConstr(u[i]>=0)
            model.addConstr(u[i]<=Vmax-1)
        model.addConstr(r[2*n]==0)
        
        
        ## Initial routes
        if ini != []:
            for k, route in enumerate(ini):
                for pos in range(length(route)-1):
                    x[route[pos], route[pos+1], k].Start = 1
                    
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
                         
                         
        # Return list of routes from solution and objective value
        if model.solCount > 0:
            routes = []
            for k in range(m):
                route = []
                current = 2*n # we start at dummy node            
                isFinished = False
                while not isFinished:
                    city = 0
                    while x[current, city, k].X == 0:
                        city+=1
                    if city == 2*n: # when we return to dummy node, we stop
                        isFinished = True
                    else:
                        route.append(city)
                        current = city
                routes.append(route)
            return routes, model.objVal, obj_time, model.Status == GRB.OPTIMAL      
        else:
            return [], 0, obj_time, False # No feasible solution found 
        

    except gp.GurobiError as e:
        print ("Error code" + str (e. errno ) + ":" + str(e))
    except AttributeError as e:
        print ("Encountered an attribute error ")
        print (e)





