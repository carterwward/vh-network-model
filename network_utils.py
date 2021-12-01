import networkx as nx
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


def generate_membership_dist(size:int, membership_dist:str="np.random.uniform(2,4)") -> np.ndarray:
    """Generates membership distribution based on a number of nodes and distribution to sample from

    Args:
        size (int): Number of nodes in the desired graph
        membership_dist (str, optional): Distribution which our # of membership stubs will be sampled from. Defaults to "np.random.uniform(1,3)".

    Returns:
        np.ndarray: Array containing available membership "stubs" identified by the node they are attatched to
    """
    mem_edges = np.array([])
    for i in range(size):
        mem_edges= np.append(mem_edges, [i for j in range(round(eval(membership_dist)))])
    np.random.shuffle(mem_edges)
    return mem_edges


def generate_size_dist(membership_edges:np.ndarray, size_dist:str="np.random.uniform(2,4)") -> dict:
    """Maps membership edges to groups based on group size sampling distribution

    Args:
        membership_edges (np.ndarray): Array containing available membership "stubs" identified by the node they are attatched to
        size_dist (str, optional): Distribution which our group sizes will be sampled from. Defaults to "np.random.uniform(1,3)". Defaults to "np.random(1,3)".

    Returns:
        dict: Map of group to array of members
    """
    group_map = {}
    group_num = 0

    while len(membership_edges) > 3:
        group_size = round(eval(size_dist))
        group_mems = []

        for i in range(group_size):
            indeces = np.where(np.in1d(membership_edges, group_mems) == False)[0]
            if len(indeces) == 0:
                continue
            group_mems.append(int(membership_edges[indeces[-1]]))
            membership_edges = np.delete(membership_edges, indeces[-1])
        group_map[group_num] = group_mems

        group_num += 1

    return group_map


def fill_graph(size:int, group_map:dict, n_prop:float=0.33, h_prop:float=0.33) -> nx.Graph:
    """Fills graph with cluster structure

    Args:
        size (int): Number of nodes in the graph
        group_map (dict): Mapping of nodes to groups
        n_prop (float, optional): probability of non-hesitant initial nodes. Defaults to 0.33.
        h_prop (float, optional): probability of hesitant initial nodes. Defaults to 0.33.

    Returns:
        nx.Graph: Filled out clustered graph
    """
    G = nx.Graph()
    for i in range(size):
        G.add_node(i)

        # TODO: For now randomly initialize state, in future may want to assign starting state based on membership and group size (prop of N and that cdf of group sizes)
        prob = np.random.uniform()
        if prob < n_prop:
            G.nodes[i]["status"] = "N"
        elif prob < n_prop + h_prop:
            G.nodes[i]["status"] = "H"
        else:
            G.nodes[i]["status"] = "U"
        G.nodes[i]["groups"] = []

    for group, members in group_map.items():
        for i in members:
            G.nodes[i]["groups"].append(group)
            for j in members:
                if i != j:
                    G.add_edge(i, j)
    return G


def viz_graph(G: nx.Graph) -> None:
    """Plots network with labeled statuses

    Args:
        G (nx.Graph): Network graph
    """
    # Color mapping
    groups = set(nx.get_node_attributes(G,'status').values())
    mapping = dict(zip(sorted(groups),count()))
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=max(mapping.values()))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    # Plt figure for legend
    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)
    for label in mapping:
        ax.plot([0],[0],color=scalarMap.to_rgba(mapping[label]),label=label)
    n_colors = [mapping[G.nodes[n]['status']] for n in G.nodes()]
    f.set_facecolor('w')
    f.tight_layout()
    plt.legend()
    plt.axis('off')

    # Draw network
    pos=nx.spring_layout(G)
    nx.draw(G,pos, with_labels=False, cmap = jet, node_size=200, node_color=n_colors)

    # Show legend
    plt.show()


def get_count_map_max(count_map:dict) -> str:
    """Returns status with highest count

    Args:
        count_map (dict): map of status to counts

    Returns:
        str: status
    """
    counts = list(count_map.values())
    max_count = max(counts)
    return np.random.choice([k for k,v in count_map.items() if v == max_count])


def process_majority(majority:str, current_status:str, G:nx.Graph, current_node:int, probability_map:dict, peer_pressure:bool=False, count_map:dict=None):
    """Given the probability, assesses whether or not given transition will be made

    Args:
        majority (str): Most influential surrounding status
        current_status (str): current node status
        G (nx.Graph): The network
        current_node (int): Identity of the current node
        probability_map (dict): Map of transitions to probabilities and new states
        peer_pressure (bool, optional): Whether or not peer pressure will be used for H to U influence. Defaults to False.
        count_map (dict, optional): Map of status to counts. Defaults to None.
    """
    p, new_status = probability_map[majority+current_status]
    # Check if we need to apply peer pressure dynamic
    if peer_pressure and majority == "H" and current_status == "U":
        # Scale influence by % of groups that were a part of majority
        p += (count_map[majority]/sum(list(count_map.values())))*(1-p) / 2

    if p > np.random.uniform():
        G.nodes[current_node]["status"] = new_status


# TODO: Change to params dictionary
def update_clustered_network(G:nx.Graph, probability_map:dict, mem_map: dict, peer_pressure:bool=True) -> dict:
    """Aggregates most influential status for each node and processes the majority.

    Args:
        G (nx.Graph): The network
        probability_map (dict): Map of transitions to probabilities and new states
        mem_map (dict): Map of groups to member lists
        peer_pressure (bool, optional): Whether or not peer pressure will be used for H to U influence. Defaults to True.

    Returns:
        dict: status mapped to population count
    """
    nodes = list(G.nodes)
    pop_counts = {"H": 0, "N": 0, "U":0}
    # Iterate over nodes in the network

    for node in nodes:
        # get status
        node_status = G.nodes[node]["status"]
        # Get group list of the node
        node_groups = G.nodes[node]["groups"]
        group_count_map = {"H": 0, "N": 0, "U":0}

        for group_num in node_groups:
            # Counts of votes from groups
            count_map = {"H": 0, "N": 0, "U":0}
            for neighbor in mem_map[group_num]:
                neighbor_status = G.nodes[neighbor]["status"]
                count_map[neighbor_status] += 1

            highest_status = get_count_map_max(count_map)
            group_count_map[highest_status] += 1

        majority = get_count_map_max(group_count_map)

        if majority != "U" and majority != node_status:
            process_majority(majority, node_status, G, node, probability_map, peer_pressure=peer_pressure, count_map=group_count_map)
        pop_counts[G.nodes[node]["status"]] += 1
    return pop_counts
