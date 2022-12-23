import numpy as np



def generateCoordinates(n, dim):
    """
        Create uniform n random coordinates in box [-dim,dim]*[-dim,dim].
    """
    coordinates = []
    for i in range(n):
        x = np.random.randint(-dim,dim+1)
        y = np.random.randint(-dim,dim+1)
        coordinates.append((x,y))
    return coordinates


def generateTimes(coordinates):
    """
        Computes travel times matrix from coordinates.
    """
    n = int(len(coordinates)/2)
    distance = np.zeros((2*n,2*n), np.float32)
    for i in range(2*n-1):
        (x_i,y_i) = coordinates[i]
        for j in range(i+1, 2*n):
            (x_j,y_j) = coordinates[j]
            distance[i][j] = np.sqrt((x_j-x_i)**2 + (y_j-y_i)**2)
            distance[j][i] = distance[i][j]
    return distance


def generateDemands(n, wMax):
    """
        Generate random uniform demand in [1, Wmax] for each request.
    """
    weight = np.zeros(2*n)
    for i in range(n):
        weight[i] = np.random.randint(1,wMax+1)
        weight[i+n] = -1*weight[i]
    return weight


def generatePDP(name, n, m, C, wMax, Vmax = 0, dim = 100):
    """
        Generate random PDP instance.
    """
    coordinates = generateCoordinates(2*n, dim)
    times = generateTimes(coordinates)
    demands = generateDemands(n, wMax)
    
    # If max nb of visits is not specified then, we compute if based on the nb of cities and the nb of vehicles
    if Vmax==0:
        Vmax = int(np.ceil(2*n/m) * 2) # *2 to allows unbalanced distribution of nb cities per route
        if Vmax % 2 == 1:
            Vmax += 1
     
    instance = {
        'name' : name,
        'n' : n,
        'm' : m,
        'times' : times.tolist(), # 2n*2n matrix --> consider distance with depot
        'coordinates' : coordinates, # coordinates of n cities + depot --> dimension n+1
        'demands' : demands.tolist(), # demands of n requests --> dimension 2*n --> demands[i] = -demands[n+i]
        'capacity' : C,
        'Vmax' : int(Vmax)
    }
    
    return instance


