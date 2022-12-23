# routing_models_algorithms

This repository gathers models and algorithms for two vehicle routing problems:
Capacitated Vehicle Routing Problem + Pickup and Delivery Vehicle Routing Problems.
Time-related constraints are not considered.

In CVRP directory, there is a parser for the instances from CVRPLib (http://vrp.galgos.inf.puc-rio.br/index.php/en/).
In PDPVRP directory, there is a parser for the instances from Li and Lim's benchmark (https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark

The MILP models require gurobipy.
Visualisation of solution requires networkx and matplotlib.

The ALNS implemented for the PDVRP is based on the following article: https://www.jstor.org/stable/25769321.










