#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Visualize routes solution on a given instance
# Each route has a different color
def plotSolution(instance, routes):
    
    coordinates = instance['coordinates']
    
    G = nx.Graph()
    
    colorMap = np.zeros(len(coordinates))
    
    # Adding city
    for city in range(len(coordinates)):
        G.add_node(city,pos=(coordinates[city][0],coordinates[city][1]))
    
    # Adding edges
    for k in range(len(routes)):
        route = routes[k]
        if route != []:
            for i in range(len(route)-1):
                colorMap[route[i]] = 2*(k+1) # different color for each route
                G.add_edge(route[i], route[i+1])
            G.add_edge(route[len(route)-1], route[0])
            colorMap[route[len(route)-1]] = 2*(k+1)
    
    colorMap[0] = 3*len(routes)
    
    # Plot
    pos = nx.get_node_attributes(G,'pos')
    fig, ax = plt.subplots(figsize=(12,12))
    nx.draw(G,pos, with_labels=True, node_size=100, font_size=10, ax=ax,node_color=colorMap.tolist(), cmap=plt.cm.Spectral)


