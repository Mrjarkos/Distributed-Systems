import numpy as np
import networkx as nx
from scipy import linalg
import matplotlib.pyplot as plt

np.random.seed(42)

def extract_complex_re_img(data):
    x = [ele.real for ele in data]
    y = [ele.imag for ele in data]
    return x, y

def plot_complex(data, title: str, fi):
    fig = plt.figure()
    x, y = extract_complex_re_img(data)
    plt.scatter(x, y)
    ## Anotate
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        plt.text(x_i, y_i, f"$\\lambda_{i}$")
    plt.grid()
    plt.title(title)
    plt.xlabel("Real")
    plt.ylabel("Img")
    return fig

# def annotate(data:tuple, text:str, fig):
    

def get_states_as_vector(G: nx.Graph):
    return np.matrix([G.nodes[node]["states"] for node in G.nodes])

def update_states(G_t: nx.Graph, Ts=200):
    # states = [nodes [states s, \dot{s}]]
    n = G_t.number_of_nodes()
    states = get_states_as_vector(G_t)
    # Protocol
    L = nx.laplacian_matrix(G_t)
    #L = nx.normalized_laplacian_matrix(G)
    # Balanceado
    s_dot = -1/Ts*L @ states[:, 0]
     
    # Update current graph
    for node, s in zip(G_t.nodes, s_dot.flat):
        G_t.nodes[node]["states"][0] += s
        G_t.nodes[node]["states"][1] = s
        
# * Agreement protocol over a triangle
# Unbalanced
edge_list = [(1, 2, 1), 
             (2, 3, 1/5),
             (3, 1, 1)]

# Balanced
edge_list = [(1, 2, 1), 
             (2, 3, 1),
             (3, 1, 1)]

n = 3
states_0 = np.array(10*np.random.rand(n)-5) # Initial state

nodes = [(i, {"states": [s, 0]}) for i, s in enumerate(states_0, 1)]
# adj_matrix = np.array([[0, 1, 1],
#           [1, 0, 1],
#           [1, 1, 0]])
# G = nx.from_numpy_array(adj_matrix)

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(edge_list)


fig = plt.figure()
pos = nx.spring_layout(G)
nx.draw(
    G, pos, edge_color='black', width=1, linewidths=1,
    node_size=500, node_color='pink', alpha=0.9,
    labels={node: node for node in G.nodes()}
)
nx.draw_networkx_edge_labels(G, pos=pos)
plt.axis("off")

# * Incidence Matric
inc_matrix = nx.incidence_matrix(G)
print(f"inc_matrix = {inc_matrix.toarray()}")

# * Adjacency Matrix
adj_matrix = nx.adjacency_matrix(G)
print(f"adj_matrix = {adj_matrix.toarray()}")

# * Laplacian Matrix
L = nx.laplacian_matrix(G)
#L = nx.normalized_laplacian_matrix(G)
print(f"laplacian_matrix = {L.toarray()}")

# * Eigenvalues/Eigenvectors
eigenvalues = sorted(linalg.eigvals(L.toarray()))
print(f"Eigenvalues = {eigenvalues}")
fig2 = plot_complex(eigenvalues, "Eigenvalues")

# left eigenvectors
# eig [0] -> Eigenvalues [1] -> Eigenvectors
# eigh -> complex Hermitian (conjugate symmetric) or a real symmetric matrix
w = linalg.eig(L.toarray(), left=True, right=False)[1]
print("Left Eigenvectors (normalized):\n", w)

# right eigenvectors
v = linalg.eig(L.toarray(), left=False, right=True)[1]
print("Right Eigenvectors (normalized):\n", v)

# * Dynamic model
T = 10
Ts = 200
logs = np.zeros((T*Ts, n, 2))
for i in range(T*Ts):
    for v, node in enumerate(G.nodes):
        logs[i][v][:] = G.nodes[node]["states"]
    
    update_states(G, Ts)
    

# * stationary state 
# Tau
tau = 1/eigenvalues[1].real
print(f"Tss={5*tau}")

# For an undirected complete graph, consensus value=weight*avg
c = np.matrix(w).transpose() @ np.matrix(states_0.reshape(n, 1))
print(f"consensus = {c}")

## Plot
fig3, ax = plt.subplots(2, 1, sharex=True)
t = np.arange(0, T, 1/Ts)
for v, node in enumerate(G.nodes):
    ax[0].plot(t, logs[:, v, 0], label = str(node))
    ax[1].plot(t, logs[:, v, 1], label = str(node))

## Annotate Cs, Tss

ax[0].legend()
ax[1].legend()
ax[0].grid()
ax[1].grid()
ax[1].set_xlabel("Tiempo (s)")
ax[0].set_ylabel("$s$")
ax[1].set_ylabel("$\\dot{s}$")

plt.show(block=False)
input("Press enter to exit")