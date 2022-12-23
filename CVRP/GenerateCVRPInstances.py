import numpy as np

# Random generation of n coordinates of locations in a box [-dim, dim]x[-dim, dim] 
# with a depot located at depotLocation.
# Also returns the distances and angle gaps among locations around the depot.
def generateCoordinates(n, dim, depotLocation):
    # Generate coordinates
    (x_d, y_d) = depotLocation
    coordinates = [depotLocation]
    for i in range(n):
        x = np.random.randint(-dim,dim)
        y = np.random.randint(-dim,dim)
        coordinates.append((x,y))
        
    # Generate angle of cities around the depot
    angle = np.zeros(n, np.float16)
    for i in range(1, n+1):
        (x_i,y_i) = coordinates[i]
        angle[i-1] = np.arctan2(y_i-y_d, x_i-x_d)
    
    # Generate angle gaps between each pair of city, excluding depot
    angleGap = np.zeros((n,n), np.float16)
    for i in range(n):
        for j in range(i+1,n):
            angleGap[i][j] = min(np.abs(angle[j]-angle[i]), 2*np.pi-np.abs(angle[j]-angle[i]))
            angleGap[j][i] = angleGap[i][j]
    
    # Generate traveling times
    times = np.zeros((n+1,n+1), np.float16)
    for i in range(0,n+1):
        (x_i,y_i) = coordinates[i]
        for j in range(i,n+1):
            x_j,y_j =coordinates[j]
            times[i][j] = np.sqrt((x_j-x_i)**2 + (y_j-y_i)**2)
            times[j][i] = times[i][j]
    
    return times, angleGap, coordinates, angle


# Random generation of demands.
# We can set the total demand with the slack parameter: totalDemands = C*m - slack
# The bigger the slack parameter, the easier the instance.
def generateDemands(n, m, C, slack):
    
    totalDemands = C*m - slack
    
    # Average demand of a customer based on the total demand (uniform distribution)
    averageDemand = int(np.round((totalDemands*(1.0/n)))) 
    
    demand = np.zeros(n, dtype=np.int16)
    sum=0
    for i in range (n):
        # The demand is randomly uniformly chosen between 1 and 2*averageDemand
        demand[i] = np.random.randint(1,2*averageDemand+1)
        sum+=demand[i]
    
    # If the actual total demand is different from expected, we increase/decrease randomly some customer demands
    # until the total demand is such as expected
    diff = totalDemands - sum
    while(diff!=0):
        idx = np.random.randint(n)
        if(diff<0 and demand[idx]>1):
            demand[idx]-=1
            diff+=1
        if(diff>0 and demand[idx]<C):
            demand[idx]+=1
            diff-=1
    return demand


# Random generation of a CVRP instance
def generateCVRP(n, m, C, slack, name, dim = 100, depotLocation=(0,0), Vmax=0):
    
    times, angleGap, coordinates, angles = generateCoordinates(n, dim, depotLocation)
    demands = generateDemands(n,m,C,slack)
    
    instance = {
        'name' : name,
        'n' : n,
        'm' : m,
        'times' : times.tolist(), # (n+1)*(n+1) matrix --> consider distance with depot
        'angleGap' : angleGap.tolist(), # n*n matrix --> does not consider angle gap with depot
        'coordinates' : coordinates, # coordinates of n cities + depot --> dimension n+1
        'angles' : angles.tolist(),
        'demands' : demands.tolist(), # demands of n cities --> dimension n
        'capacity' : C,
        'Vmax' : Vmax
    }
    
    # If max nb of visits is not specified then, we compute if based on the maximum number of demands a vehicle can carry
    if Vmax == 0:
        d = sorted(instance["demands"])
        idx= 0
        cumul = 0
        while d[idx] + cumul <= C:
            cumul += d[idx]
            idx += 1
        instance['Vmax'] = idx
        
    
    return instance



