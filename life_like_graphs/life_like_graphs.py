import argparse
import unittest

import numpy as np
#import matplotlib.pyplot as plt

gaussian = lambda x: np.exp(- (x**2 / 0.5**2) / 2)  

def get_ca_mlp(birth=[3], survival=[2,3]):
    """ 
    return an MLP forward pass function encoding Life-like CA rules
    default to Conway's Game of Life (B3/S23)
    """ 

    wh = np.ones((18,1))
    bh = -np.arange((18)).reshape(18,1)
    wy = np.zeros((1,18))

    for bb in birth:
        wy[:, bb] = 1.0 

    for ss in survival:
        wy[:, ss+9] = 1.0 

    def mlp(x):

        hidden = gaussian(np.dot(wh, x) + bh)
        out = np.round(np.dot(wy, hidden))

        return out 

    return mlp 

def ca_graph(length):
    
    # nodes
    # number of nodes is the side of the grid, squared.
    num_nodes = length**2
    nodes = np.zeros((num_nodes, 1))
    node_indices = np.arange(num_nodes)

    # edges
    num_edges = 8 * num_nodes
    edges = np.zeros((num_edges, 1))
    
    # senders & receivers
    senders = np.vstack(\
            [node_indices - length -1, \
            node_indices - length, \
            node_indices - length + 1, \
            node_indices - 1, \
            node_indices + 1, \
            node_indices + length - 1, \
            node_indices + length, \
            node_indices + length + 1])
    sender = senders.T.reshape(-1)

    senders = (senders + length**2) % length**2
    receivers = np.repeat(node_indices, 8)

    return (num_nodes, num_edges, nodes, edges, senders, receivers)

def add_glider(graph_tuple):

    nodes = graph_tuple[2]
    length = int(np.sqrt(graph_tuple[0]))

    nodes[0, 0] = 1.0
    nodes[1, 0] = 1.0  
    nodes[2, 0] = 1.0
    nodes[2 + length, 0] = 1.0
    nodes[1 + 2 * length, 0] = 1.0

    return (graph_tuple[0], nodes, graph_tuple[2], \
            graph_tuple[3], graph_tuple[4], graph_tuple[5])


def get_adjacency(graph_tuple):

    num_nodes = graph_tuple[0]
    num_edges = graph_tuple[1]
    length = int(np.sqrt(graph_tuple[0]))
    
    senders = graph_tuple[4]
    receivers = graph_tuple[5]

    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    
    for xx in range(receivers.shape[0]):

        for yy in senders[:, receivers[xx]]:
            adjacency_matrix[receivers[xx], yy] = 1.0

    return adjacency_matrix

def get_graph_grid(graph_tuple):

    length = int(np.sqrt(graph_tuple[0]))

    grid = np.zeros((length, length))

    
    for ii, node_state in enumerate(graph_tuple[2]):

        grid[ ii // length, ii % length] = node_state

    return grid


        
if __name__ == "__main__":

    graph_tuple = ca_graph(5)   

    grid = get_graph_grid(graph_tuple)

    graph_tuple = add_glider(graph_tuple)
    grid = get_graph_grid(graph_tuple)

    a_matrix = get_adjacency(graph_tuple)

    print(grid) 
    mlp = get_ca_mlp()


    gt = graph_tuple
    for ii in range(10):
        
        new_nodes = gt[2]

        
        new_nodes = mlp(((a_matrix @ new_nodes) + 9 * new_nodes).T).T

        gt = (gt[0], gt[1], new_nodes, gt[3], gt[4], gt[5])

        grid = get_graph_grid(gt)
        print(grid)
        
        import pdb; pdb.set_trace()



