import re
import glob
import scipy
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import os

from collections import defaultdict

from copy import deepcopy
import copy
from sklearn.linear_model import LinearRegression

from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from sklearn import manifold

from scipy.stats import pearsonr


def get_links(precision_matrix_df, threshold=0.02):
    """
    Creates a DataFrame of links between nodes based on the provided precision matrix.

    Parameters
    ----------
    precision_matrix_df : pd.DataFrame
        A DataFrame that consists of 3 columns: var1, var2, and weight.
    threshold : float
        Threshold for inverse covariance estimate to be considered as non-zero. Allows to filter out very weak connections. Default is 0.2.

    Returns
    -------
    links : pd.DataFrame
        A DataFrame that consists of 3 columns: var1, var2, and weight.
    """
    precision_matrix_df = precision_matrix_df.where(np.triu(np.ones(precision_matrix_df.shape)).astype(bool))

    links = precision_matrix_df.stack().reset_index()
    links.columns = ['var1', 'var2','weight']
    links=links.loc[ (abs(links['weight']) > threshold) &  (links['var1'] != links['var2']) ]
    links = links.round(3)
    
    return links

def leave_n_edges_from_eeg(links, N=2):
    """
    Restricts links in the graph to N links distant from links containing eeg.

    Parameters
    ----------
    links : pd.DataFrame
        A DataFrame that consists of 3 columns: var1, var2, and weight.
    N : int
        Distance from EEG nodes.

    Returns
    -------
    edges_df : pd.DataFrame
        A DataFrame that consists of 3 columns: var1, var2, and weight.
    """
    
    parents = set(links[links['var1'].str.contains("e_ERN") | links['var1'].str.contains("e_CRN")]['var1'])

    final_edges = []
    edges_df = pd.DataFrame({})
    leafs = set()

    for parent in parents:
        this_leafs = []

        parent_edges = links[links['var1'] == parent]
        final_edges.append(parent_edges)
        leafs.update(parent_edges['var2'])


    for i in range(1,N+1):
        parents = leafs
        leafs = set()

        for parent in parents:
            parent_edges = links[(links['var1'] == parent) | (links['var2'] == parent)]

            final_edges.append(parent_edges)
            leafs.update(parent_edges['var2'])
            leafs.update(parent_edges['var1'])

    edges_df = pd.concat(final_edges, axis=0, ignore_index=True)
    edges_df = edges_df.drop_duplicates(subset = ["var1", 'var2'])
    
    return edges_df

def draw_graph(
    links, 
    basic_links=None, 
    basic=False, 
    mapping=None, 
    seed=1, 
    nodes_predictabilities=None,
    node_colors = None,
    scale=1, 
    specific_positions = [], 
    layout = None,
    nx_layout=None,
    save=None,
    dir=None,
):
    """
    Draws graph based on the provided links DataFrame

    Parameters
    ----------
    links : pd.DataFrame
        A DataFrame that consists of 3 columns: var1, var2, and weight
    basic_links : pd.DataFrame, optional
        A link DataFrame for links to be drawn in a special way. Default is None.
    basic : bool or str, optional
        Indicates how to draw basic_links. 
        'first' draws basic_links bolded to make them more distinct.
        'only' draws only basic_links. Default is False.
    mapping : dict, optional
        Dictionary where the keys are nodes names from links DataFrame, and values are the node names to be drawn on the graph.
        Default is None.
    seed : int, optional
        Seed to be passed to nx.spring_layout. Default is 1.
    nodes_predictabilities: pd.DataFrame, optional
        Predictabilities to be drawn around the nodes. The DataFrame has nodes as columns and predictabilities ac values. Default is None.
    scale : float, optional
        Scale to be passed to nx.spring_layout. Default is 1.
    specific_positions : List[Tuple[str, List[float]]], optional
        Set positions of provided nodes. List of tuples in a form: [('node_name', [x, y])]. Default is [].
    layout : List[Tuple[str, List[float]]], optional
        Layout of the graph. Must contain positions of all nodes in the graph. If provided, it replaces the default nx.spring_layout.
        Default is None.
    save : str or None, optional
        Whether to save graph figure. If a string is provided, it is assumed to be a path where to save the figure. Default is None.

    Returns
    -------
    pos : List[Tuple[str, List[float]]]
        Layout of the graph.
    G: nx.Graph
        Graph created based on the provided links
    """
    
    # set plotting parameters
    cm = 1/2.54
    dpi = 700
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams["axes.edgecolor"] = ".15"
    plt.rcParams["axes.linewidth"]  = 0.5
    sns.set_style("white")
    palette = sns.color_palette("colorblind")
    
          
    fig = plt.figure(3, figsize=(10*cm, 10*cm))
    ax = fig.add_axes([0,0,1,1], aspect=1)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    
    # create graph
    G = nx.from_pandas_edgelist(links,'var1','var2', edge_attr='weight', create_using=nx.Graph())
    G = nx.relabel_nodes(G, mapping) if mapping is not None else G
    
    # set nodes colors
    node_color = [node_colors[node] if node in node_colors else palette[0] for node in G.nodes()]
    # set nodes position
    edges = G.edges() 
    
    for u,v in edges:
        G[u][v]['weight'] = abs(G[u][v]['weight'])
    
    if (layout == None) & (nx_layout == None):
        pos_i = nx.spring_layout(G, k=60*(1/np.sqrt(len(G.nodes()))), iterations=1000, scale=scale, seed=seed)
        pos = pos_i
        # set specific positions, if desired
        for position in specific_positions:
            pos[position[0]] = position[1]
    elif (layout != None) & (nx_layout == None):
        pos = layout
    else:
        pos = nx_layout(G, scale=scale)

    # if basic is False, draw all links
    if basic is False:
        G = nx.from_pandas_edgelist(links,'var1','var2', edge_attr='weight', create_using=nx.Graph())
        G = nx.relabel_nodes(G, mapping) if mapping is not None else G
        
        edges = G.edges() 
           
        weights = []
        for u,v in edges:
            weight = G[u][v]['weight']
            # print(f"{u},{v},{weight}")
            weights.append(weight)
            
        edge_color = [('red' if edge < 0 else 'blue') for edge in weights]

        weights_alphas = []
        weight_bold = []
        
        # set the width of links based on the weights
        for edge_weight in weights:
            if abs(edge_weight) < 0.05:
                alpha = 0.2 
                bold = 1
            elif abs(edge_weight) >= 0.05 and abs(edge_weight) < 0.1:
                alpha = 0.25
                bold = 1
            elif abs(edge_weight) >= 0.1 and abs(edge_weight) < 0.15:
                alpha = 0.3
                bold = 2
            elif abs(edge_weight) >= 0.15 and abs(edge_weight) < 0.2:
                alpha = 0.35
                bold = 1
            elif abs(edge_weight) >= 0.15 and abs(edge_weight) < 0.2:
                alpha = 0.38
                bold = 1
            elif abs(edge_weight) >= 0.2 and abs(edge_weight) < 0.25:
                alpha = 0.40
                bold = 1
            elif abs(edge_weight) >= 0.25 and abs(edge_weight) < 0.3:
                alpha = 0.42
                bold = 1
            elif abs(edge_weight) >= 0.3 and abs(edge_weight) < 0.35:
                alpha = 0.45
                bold = 1
            else:
                alpha = 0.5
                bold = 1
            weights_alphas.append(alpha)
            weight_bold.append(bold)

    # if basic is not False, draw links provided as 'basic_links'  using one of the following startegies:'first' | 'only'
    if (basic is not False):
        G = nx.from_pandas_edgelist(links,'var1','var2', edge_attr='weight', create_using=nx.Graph())
        G = nx.relabel_nodes(G, mapping) if mapping is not None else G
        
        edges = G.edges() 
           
        weights = []
        for u,v in edges:
            weight = G[u][v]['weight']
            weights.append(weight)
            
        edge_color = [('red' if edge < 0 else 'blue') for edge in weights]
        
        weights_alphas = []
        weight_bold = []

        for edge_weight in weights:
            alpha = 0.15 
            bold = 1
            weights_alphas.append(alpha)
            weight_bold.append(bold)
            
        ###########################################################################################################
        G_eeg = nx.from_pandas_edgelist(basic_links,'var1','var2', edge_attr='weight', create_using=nx.Graph())
        G_eeg = nx.relabel_nodes(G_eeg, mapping) if mapping is not None else G_eeg
        
        eeg_edges = G_eeg.edges() 
        
        weights_eeg = []
        for u,v in eeg_edges:
            weight = G_eeg[u][v]['weight']
            print(f"{u},{v},{weight}")
            weights_eeg.append(weight)
            
        edge_eeg_color = [('red' if edge < 0 else 'blue') for edge in weights_eeg]
        
        weights_eeg_alphas = []
        weights_eeg_bold = []

        for edge_weight in weights_eeg:
            if abs(edge_weight) < 0.05:
                alpha = 0.3 
                bold = 2
            elif abs(edge_weight) >= 0.05 and abs(edge_weight) < 0.09:
                alpha = 0.35
                bold = 2
            elif abs(edge_weight) >= 0.09 and abs(edge_weight) < 0.16:
                alpha = 0.40
                bold = 3
            elif abs(edge_weight) >= 0.16 and abs(edge_weight) < 0.5:
                alpha = 0.45
                bold = 3
            else:
                alpha = 0.6
                bold = 3
            weights_eeg_alphas.append(alpha)
            weights_eeg_bold.append(bold)
            
        
    nx.draw_networkx_nodes(
        G, 
        pos=pos,
        linewidths=0.5,
        edgecolors='black',
        node_size = 205,
        node_color=node_color,
    )

    nx.draw_networkx_labels(
        G, 
        pos=pos,
        font_size=6,
    )
    
    if basic == False:
        nx.draw_networkx_edges(
            G, 
            pos=pos,
            edgelist = G.edges(),
            edge_color = edge_color,
            alpha=weights_alphas,
            width = weight_bold,
        )
        

    if basic_links is not None and basic == 'only':
        nx.draw_networkx_edges(
            G, 
            pos=pos,
            edgelist = G_eeg.edges(),
            edge_color = edge_eeg_color,
            alpha=weights_eeg_alphas,
            width = weights_eeg_bold,
        )
    
    if basic_links is not None and basic == 'first':
        nx.draw_networkx_edges(
            G, 
            pos=pos,
            edgelist = G.edges(),
            edge_color = edge_color,
            alpha=weights_alphas,
            width = weight_bold,
        )
        
        nx.draw_networkx_edges(
            G, 
            pos=pos,
            edgelist = G_eeg.edges(),
            edge_color = edge_eeg_color,
            alpha=weights_eeg_alphas,
            width = weights_eeg_bold,
        )
    
    # add nodes predictabilities
    if nodes_predictabilities is not None:
        nodes_predictabilities = nodes_predictabilities.rename(columns=mapping) if mapping is not None else nodes_predictabilities

        patches = []
        colors = []
        for item in pos.items():
            node_id = item[0]
            node_predictability = nodes_predictabilities[node_id].to_list()[0]
            node_predictability_percent = 360 * node_predictability
            cor_x, cor_y = item[1]

            weg_bck = Wedge((cor_x, cor_y), .065, 0, 360, width=0.03, edgecolor='black', linewidth=0.3, facecolor='white')
            patches.append(weg_bck)
            colors.append(0)
            ax.add_patch(weg_bck)

            weg = Wedge((cor_x, cor_y), .065, 0, node_predictability_percent, width=0.03, edgecolor='black', linewidth=0.3, facecolor='gray')
            patches.append(weg)
            colors.append(1)
            ax.add_patch(weg)

    plt.show()

    if save != None:
        fig.savefig(os.path.join(dir, f'{save}.png'), bbox_inches='tight')

    return pos, G

def calculate_nodes_predictability(
    X, 
    precision_matrix_df, 
    threshold=0.02
):
    """
    Calculates nodes' predictability based on the provided dataset. It uses the provided dataset to make regression for 
    each node with independent variables being the neighbors defined by precision_matrix_df.
    
    Parameters
    ----------
    X : pd.DataFrame
        A dataset of shape (n_samples, n_features).
    precision_matrix_df : pd.DataFrame
        A DataFrame representing the precision matrix, typically derived from a graphical model.
    threshold : float, optional
        Threshold for the inverse covariance estimate to be considered as non-zero. Allows filtering out very weak connections. 
        Default is 0.02.
    
    Returns
    -------
    explained_variance_df : pd.DataFrame
        A DataFrame with nodes as columns and nodes' predictability as values.
    """
    
    precision_matrix_df_mask = precision_matrix_df.mask(abs(precision_matrix_df) <= threshold, False)
    precision_matrix_df_mask = precision_matrix_df_mask.mask(abs(precision_matrix_df_mask) > threshold, True)

    # Set the diagonal elements to False
    for i in range(len(precision_matrix_df_mask)):
        precision_matrix_df_mask.iat[i, i] = False  

    explained_variance = []

    for node in precision_matrix_df_mask.columns:
        mask = precision_matrix_df_mask.loc[node].to_numpy()
        masked_colums = precision_matrix_df_mask.columns[mask]    
        
        y_ = X[[node]]
        X_ = X[masked_colums]

        lm = LinearRegression()
        lm.fit(X_, y_)

        score = lm.score(X_,y_)
        explained_variance.append(score)

    explained_variance_df = pd.DataFrame(np.array(explained_variance).reshape(1,-1), columns=precision_matrix_df.columns)
    return explained_variance_df



def manifold_position(df, node_list, mapper):
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver="dense", n_neighbors=4
    )
    
    df_ = df.rename(columns = mapper)
    
    df_ = df_[node_list]
    
    X = df_.to_numpy()
    X -= X.mean(axis=0) # data is centered in the model (assume_centered=False)
    X /= X.std(axis=0)

    embedding = node_position_model.fit_transform(X.T).T
    
    pos = dict()
    for index, node in enumerate(node_list):
        pos[node] = [embedding[0][index]*5, embedding[1][index]*5]
    
    return pos

def estimate_difference_graph(
    ern_graph, 
    crn_graph, 
    threshold=0.035
):
    """
    Estimates the difference graph between the provided ERN graph and CRN graph.

    Parameters
    ----------
    ern_graph : nx.Graph
        The ERN graph.
    crn_graph : nx.Graph
        The CRN graph to be subtracted from the ERN graph.
    threshold : float, optional
        Threshold for the difference between edges to be considered as non-zero. This prevents false differences between graphs. 
        Default is 0.035.
    
    Returns
    -------
    diff_graph : nx.Graph
        The difference graph showing changes between ERN and CRN graphs.
    colors : List
        List of colors of edges in the difference graph indicating the direction of association.
        Blue for negative changes, red for positive changes.
    linestyle : List
        List of linestyles of edges in the difference graph indicating the nature of the change in associations.
        Solid for edges present in the ERN graph but not in the CRN graph, dotted for associations that dropped in strength,
        dashed for associations that increased in strength.
    """
    colors = []
    linestyle = []
    remove = []

    diff_graph = ern_graph.copy()
    ern_edges = ern_graph.edges()


    for u,v in ern_edges:
        if u not in crn_graph.nodes():
            # zero links that does not exceed link threshold 
            if abs(ern_graph[u][v]['weight']) <= 0.02: 
                print(f"{u},{v},{ern_graph[u][v]['weight']} unique ERN but too low")
                diff_graph.remove_edge(u, v)
            else:
                this_color = 'red' if ern_graph[u][v]['weight'] < 0 else 'blue'
                colors.append(this_color)
                linestyle.append('solid')
        else:
            # link did not exist in CRN (so there is no change)
            if (v not in crn_graph[u].keys()): 
                # zero links that does not exceed link threshold 
                if abs(ern_graph[u][v]['weight']) <= 0.02: 
                    print(f"{u},{v},{ern_graph[u][v]['weight']} unique ERN but too low")
                    diff_graph.remove_edge(u, v)
                else:
                    this_color = 'red' if ern_graph[u][v]['weight'] < 0 else 'blue'
                    colors.append(this_color)
                    linestyle.append('solid')

            elif (ern_graph[u][v]['weight'] < 0) & (crn_graph[u][v]['weight'] < 0):
                diff = ern_graph[u][v]['weight'] - crn_graph[u][v]['weight']

                if abs(diff) <= threshold:
                    colors.append('purple')
                    linestyle.append('solid')
                else:
                    diff_graph[u][v]['weight'] = diff
                    colors.append('red')
                    this_linestyle = (0,(2, 1)) if diff < 0 else (0,(.5, .5))
                    linestyle.append(this_linestyle)
                    print(f"{u},{v},{ern_graph[u][v]['weight']} enlarged/dropped red")

            elif (ern_graph[u][v]['weight'] > 0) & (crn_graph[u][v]['weight'] > 0):
                diff = ern_graph[u][v]['weight'] - crn_graph[u][v]['weight']

                if abs(diff) <= threshold:
                    colors.append('purple')
                    linestyle.append('solid')

                else:
                    diff_graph[u][v]['weight'] = diff
                    colors.append('blue')
                    this_linestyle = (0,(.5, .5)) if diff < 0 else (0,(2, 1))
                    linestyle.append(this_linestyle)
                    print(f"{u},{v},{ern_graph[u][v]['weight']} enlarged/dropped blue")

            elif (ern_graph[u][v]['weight'] > 0) & (crn_graph[u][v]['weight'] < 0):
                colors.append('blue')
                diff = ern_graph[u][v]['weight'] - crn_graph[u][v]['weight']
                diff_graph[u][v]['weight'] = diff
                # link change the sign; the color indicate ERN sign
                this_linestyle = (0,(.5, .5)) 
                linestyle.append(this_linestyle)


            elif (ern_graph[u][v]['weight'] < 0) & (crn_graph[u][v]['weight'] > 0):
                colors.append('red')
                diff = ern_graph[u][v]['weight'] - crn_graph[u][v]['weight']
                diff_graph[u][v]['weight'] = diff
                 # link change the sign; the color indicate ERN sign
                this_linestyle = (0,(.5, .5))
                linestyle.append(this_linestyle)
        
    return diff_graph, colors, linestyle

def draw_difference_graph(
    G, 
    colors, 
    linestyle='solid', 
    no_purple = False, 
    seed=1, 
    scale=1, 
    k=1, 
    specific_positions=[], 
    layout = None, 
    save = None,
    node_colors = None,
    nx_layout=None,
    dir=None,
):
    """
    Draws graph based on the provided links DataFrame

    Parameters
    ----------
    G : nx.Graph
        The difference graph to be drawn.    
    colors : List
        List of edges colors to be passed to nx.draw_networkx_edges
    linestyle: List
        List of edges linestyles to be passed to nx.draw_networkx_edges
    no_purple : bool, optional
        Whether to draw edges that disappeard. If True, these edges are drawn in purple.
        Default is True.
    seed : int, optional
        Seed to be passed to nx.spring_layout. Default is 1.
    scale : float, optional
        Scale to be passed to nx.spring_layout. Default is 1.
    k: int, optional
         Scaling parameter to be passed to nx.spring_layout. Default is 1.
    specific_positions : List[Tuple[str, List[float]]], optional
        Set positions of provided nodes. List of tuples in a form: [('node_name', [x, y])]. Default is [].
    layout : List[Tuple[str, List[float]]], optional
        Layout of the graph. Must contain positions of all nodes in the graph. If provided, it replaces the default nx.spring_layout.
        Default is None.
    save : str or None, optional
        Whether to save graph figure. If a string is provided, it is assumed to be a path where to save the figure. Default is None.
    """
    
    cm = 1/2.54
    dpi = 500

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams["axes.edgecolor"] = ".15"
    plt.rcParams["axes.linewidth"]  = 0.5
    sns.set_style("white")
    palette = sns.color_palette("colorblind")
    
    fig = plt.figure(3, figsize=(10*cm, 10*cm))
    axes = fig.add_axes([0,0,1,1], aspect=1)
    axes.set_xlim([-1,1])
    axes.set_ylim([-1.07,1.3])
    
    colors_ = colors
    linestyle_ = linestyle
    
    G_copy = G.copy()
    edges = G_copy.edges()
        
    for u,v in edges:
        G_copy[u][v]['weight'] = abs(G_copy[u][v]['weight'])
    
    if (layout == None) & (nx_layout == None):
        pos_i = nx.spring_layout(G, k=60*(1/np.sqrt(len(G.nodes()))), iterations=1000, scale=scale, seed=seed)
        pos = pos_i
        # set specific positions, if desired
        for position in specific_positions:
            pos[position[0]] = position[1]
    elif (layout != None) & (nx_layout == None):
        pos = layout
    else:
        pos = nx_layout(G, scale=scale)

    ####################################################################################33
    
    if no_purple:
        edges = G.edges()
        for index, edge in enumerate(edges):
            if colors[index] == 'purple':
                G.remove_edge(edge[0], edge[1])
        G.remove_nodes_from(list(nx.isolates(G)))
        
        purple_indexes = [i for i, e in enumerate(colors) if e == 'purple']
        colors_ = [i for j, i in enumerate(colors) if j not in purple_indexes]
        linestyle_ = [i for j, i in enumerate(linestyle) if j not in purple_indexes]
    
    G_copy = G.copy()
    edges = G_copy.edges()
        
    for u,v in edges:
        G_copy[u][v]['weight'] = abs(G_copy[u][v]['weight'])
    
    # set nodes colors
    node_color = [node_colors[node] if node in node_colors else palette[0] for node in G.nodes()]
            
    edges = G.edges()
    
    
    weights = []
    for u,v in edges:
        weight = G[u][v]['weight']
        weights.append(weight)

    edge_color = colors_

    weights_alphas = []
    weight_bold = []

    for index, edge_weight in enumerate(weights):
        if colors_[index] == 'purple':
            alpha = 0.25
            bold = 1.2
        elif abs(edge_weight) < 0.05:
            alpha = 0.4 
            bold = 2
        elif abs(edge_weight) >= 0.05 and abs(edge_weight) < 0.09:
            alpha = 0.45
            bold = 2
        elif abs(edge_weight) >= 0.09 and abs(edge_weight) < 0.16:
            alpha = 0.5
            bold = 3
        elif abs(edge_weight) >= 0.16 and abs(edge_weight) < 0.5:
            alpha = 0.45
            bold = 3
        else:
            alpha = 0.6
            bold = 3
        weights_alphas.append(alpha)
        weight_bold.append(bold)
    
    
    nx.draw_networkx_nodes(
        G, 
        pos=pos,
        linewidths=0.5,
        edgecolors='black',
        node_size = 200,
        node_color=node_color,
    )

    nx.draw_networkx_labels(
        G, 
        pos=pos,
        font_size=6,
    )
    
    nx.draw_networkx_edges(
        G, 
        pos=pos,
        edgelist = G.edges(),
        edge_color = edge_color,
        style = linestyle_,
        alpha=weights_alphas,
        width = weight_bold,
    )
    
    
    plt.show() 
    
    if save != None:
        fig.savefig(os.path.join(dir, f'{save}.png'), bbox_inches='tight')