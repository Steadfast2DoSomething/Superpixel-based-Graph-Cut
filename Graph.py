import networkx as nx

def GraphCut(factor):
    G = nx.DiGraph()

    '''
    ## example:
    https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.flow.minimum_cut.html
    # x srouce node
    # y sink node
    G.add_edge('x','a', capacity = 3.0)
    G.add_edge('x','b', capacity = 1.0)
    G.add_edge('a','c', capacity = 3.0)
    G.add_edge('b','c', capacity = 5.0)
    G.add_edge('b','d', capacity = 4.0)
    G.add_edge('d','e', capacity = 2.0)
    G.add_edge('c','y', capacity = 2.0)
    G.add_edge('e','y', capacity = 3.0)
    
    cut_value, partition = nx.minimum_cut(G, 'x', 'y')
    cut_value, partition = nx.minimum_cut(G, 'x', 'y')
    '''
