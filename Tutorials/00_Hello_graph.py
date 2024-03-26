import networkx as nx
import matplotlib.pyplot as plt

# * Graph -> Any hashable object -> str, xml obj, tuple, etc
G = nx.Graph() # Unidrected graph

# ? Create Graph

# ! Add Nodes

# * 1) Add one node at the time
G.add_node(1)
# * 2) Or add nodes from iterable objects 
# ** 2.1) List
G.add_nodes_from([2, 3])
# ** 2.2) Tuple, dict (node, node_attribute_dict)
G.add_nodes_from([
    (4, {"color": "red"}),
    (5, {"color": "green"}),
])
# * 3) Nodes from other graph
H = nx.path_graph(10)
G.add_nodes_from(H)
# * 3.1) Other graph as a node
#G.add_node(H)
# ! Add Edges

# * 1) One single edge at the time
G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)  # unpack edge tuple*

# * 2) List of edges
G.add_edges_from([(1, 2), (1, 3)])

# * 3) ebunch ->any iterable container of edge-tuple ex. (2, 3, {'weight': 3.1415})
G.add_edges_from(H.edges)

print(G.number_of_nodes())
print(G.number_of_edges())


# ? Order 
# The order of adjacency reporting (e.g., G.adj, G.successors, G.predecessors) is the order of edge addition.
# However, the order of G.edges is the order of the adjacencies which includes both the order of the nodes and each node’s adjacencies.
# See example below:
DG = nx.DiGraph()
DG.add_edge(2, 1)   # adds the nodes in order 2, 1
DG.add_edge(1, 3)
DG.add_edge(2, 4)
DG.add_edge(1, 2)
print(list(DG.successors(2))) # [1, 4]
print(list(DG.edges)) #  [(2, 1), (2, 4), (1, 3), (1, 2)]

# ? Show Graph
nx.draw_spring(G, with_labels=True)
plt.show()