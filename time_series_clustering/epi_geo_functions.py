#!/usr/bin/env python3

########################
# CODE FOR Running Community Structure Detection
#
# Code written by Shweta Bansal
# LAST UPDATED: Dec 22, 2022 
########################

###########################################################
# IMPORT NECESSARY MODULES

import pandas as pd
import csv
import networkx as nx
import itertools 
import numpy as np
import math
import operator
from collections import Counter
#import importlib

# import graphing tools
import matplotlib.pyplot as plt
import plotly.offline as plto
plto.init_notebook_mode(connected=True)

# import functions for mapping
import plotly.express as px
from urllib.request import urlopen
import json
import kaleido # to save map files


# import stats/machine learning tools
import scipy as sy
from sklearn.metrics.cluster import normalized_mutual_info_score

# import community # this is the networkx implementation of the Louvain algorithm
# from https://github.com/taynaud/python-louvain
import community

# import an even faster implementation of Louvain
# SB tested this on Feb 17, 2019 and it gives identical results to community in a much shorter time
# https://louvain-igraph.readthedocs.io/en/latest/reference.html
import louvain
import igraph as ig

# import community adapted to spectral null from MacMahon et al, 2015
# Code base is from https://github.com/taynaud/python-louvain
#from community_with_spectralnull import community_louvain as community_spectralnull
#importlib.reload(community_spectralnull)

# stop warnings from printing out (comment out if debugging) 
import warnings as wn
wn.filterwarnings("ignore")

####################################################################
# PART 1:
# DATA CLEANING AND BASIC DATA ISOLATION FUNCTIONS
####################################################################

    
####################################################
def calc_timeseries_correlations(all_county_df):
# Makes a dataframe OF THE CORRELATIONS BETWEEN ALL TIMESERIES IN THE DATA
# inputs: the dataframe of county timeseries(rows: weeks, columns: counties)
# outputs: a dataframe of the correlations in the flu season county x county with 1 down diagonal
#          and the dataframe of county timeseries
   
    
    # calculate pearsons correlation between time series for each pair of time series
    # and make NaNs and negative correlations 0 (these 0s will later be ignored)
    correlation_df = all_county_df.corr(method='pearson')
    correlation_df = correlation_df.fillna(0)
    correlation_df = correlation_df.clip(lower=0) # make negative values 0
    
    return correlation_df
    
    
####################################################
def create_network(correlation_df, threshold, data, state=None):
# CREATES A NETWORK FROM THE CORRELATION DATAFRAME
# INPUTS: needs the correlation df, threshold for edgeweight to be included in the network 
# outputs: a graph of the correlation network 
    
    if state:
        list_counties = data.state.unique()
    else:
        list_counties = data.county.unique()

    matrix_1 = sy.sparse.lil_matrix(correlation_df) # convert correlation dataframe to matrix
    matrix_1.setdiag(0)                             # ignore self correlations (set the diagonals = 0)
    matrix_1[matrix_1 < threshold] = 0              # set anything below the threshold = 0 
    
    # Construct the network
    # Note: no edges exist between counties where correlation < threshold
    network = nx.from_scipy_sparse_matrix(matrix_1)
     
    # Relabel nodes with FIPS 
    nodelabel_mapping = dict(zip(range(0,len(list_counties)), list_counties))
    nx.relabel_nodes(network,nodelabel_mapping, copy=False)
	
	# Remove edges that have NaN weight 
    to_remove = [(i,j) for i,j,d in network.edges(data=True) if math.isnan(float(d['weight']))]
    network.remove_edges_from(to_remove)
    
    return network

##################################################
# Creates nearest neighbor graph for US counties (neighbor list is based on shared borders)
def make_nn_network(fullpath, state=None):
    
    # read neighbor lists
    if state:
        reader_data = csv.reader(open(fullpath+'county_neighbors_fips.txt'), delimiter=',')
        neighbor_list = [(int(int(line[0])/1000), int(int(line[1])/1000)) for line in reader_data]
    else:
        reader_data = csv.reader(open(fullpath+'county_neighbors_fips.txt'), delimiter=',')
        neighbor_list = [(int(line[0]), int(line[1])) for line in reader_data]
 
    G_nn = nx.Graph()
    for u,v in neighbor_list:
        G_nn.add_edge(u,v, weight=1)
              
    return G_nn


##################################################
# PART 2: RUN LOUVAIN COMMUNITY DETECTION ANALYSIS
##################################################

####################################################
def louvain_community_detection(network, G_nn, implementation, Cg = None, nodelabel_map=None):
# Runs Louvain community detection algorithm on weighted network
# If using 'louvain_spectralnull', need to provie Cg matrix

    if implementation == 'louvain_community':
        partition = community.best_partition(network, weight='weight')
        
    elif implementation == 'louvain_igraph': # fastest
        # convert to igraph
        edges = [(e1,e2,a['weight']) for e1,e2, a in network.edges(data=True)]
        graph_ig = ig.Graph.TupleList(edges, directed=False, weights=True, vertex_name_attr='name')
        ws = [e["weight"] for e in graph_ig.es()]
        part = louvain.find_partition(graph_ig, louvain.ModularityVertexPartition, weights = ws)
        # convert result to dictionary
        partition = {}
        for comm_id,comm_members in enumerate(part):
            comm_member_names = [graph_ig.vs[c]["name"] for c in comm_members]
            partition.update({c: comm_id for c in comm_member_names})
    
    #elif implementation == 'louvain_spectralnull':
    # code not complete, need to adapt method for induced graph (Dec 2022)
        #partition = community_spectralnull.best_partition(graph=network, Cg=Cg, nodelabel_map=nodelabel_map, weight='weight')


    # make sure all nodes are in partition (if any missing, add as NaNs)
    m1 = [n for n in list(network.nodes()) if n not in partition.keys()]   
    m2 = ([n for n in list(G_nn.nodes()) if n not in partition.keys()])
    missing_nodes = m1 + m2
    missing_nodes = list(set(missing_nodes))
    partition.update({m: float('NaN') for m in missing_nodes})

    return partition


####################################################
def reduce_num_communities(partition, thresh):
# REDUCES COMMUNITIES SMALLER THAN THRESH BY REMOVING THEM, and relabel modularity class to be consecutive
# Partition is dictionary with key: nodeid, value: community id
   
	# find the community sizes of each community
    uniq_comms = list(set(partition.values()))
    uniq_comms = [u for u in uniq_comms if float(u) != float('NaN')]
    comm_size = {i:len([node for node, part in partition.items() if part == i]) for i in uniq_comms} #dictionary
    
    # find all the communties that are too small (i.e smaller than or equal to thresh nodes)
    small_comm_id = [comm for comm,csize in comm_size.items() if (csize <= thresh)]
    
    # for communities identified as "small", make all nodes have NaN as modularity class
    for node in [key for key,val in partition.items() if val in small_comm_id]:
        partition[node] = float('NaN')
        
    # relabel modularity classes so that they are consecutive
    current_modclass = list(set(partition.values()))
    current_modclass = [int(m) for m in current_modclass if str(m) != 'nan']
    new_labels = list(range(0,len(current_modclass))) # create a new set of labels for the modularity classes
    label_dict = dict(zip(current_modclass, new_labels))
    
    # convert partition dictionary to dataframe; switch labels using map;
    part_df = pd.DataFrame.from_dict(partition, orient='index').reset_index()
    part_df = part_df.rename(columns = {'index': 'node', 0: 'modularity_class'})
    part_df['modularity_class'] = part_df['modularity_class'].map(label_dict) # relabel old labels with new labels
        
    return part_df
    

##################################################
def comm_struct_robustness(G, G_nn, implementation, num_bootstrap = 10, Cg=None, nodelabel_map=None):
# This function calculates a robust community structure 
# uses method of Donker/Wallinga/Slack/Grundmann "Hospital Networks and the Dispersal of Hospital-Acquired Pathogens by Patient Transfer"
# works by generating a bootstrap network with a different set of edge weights
# and re-computing the community structure on that graph
# finally the partition which has the most agreement to other partitions is chosen
# SB tested this method on Feb 17, 2019 and found that each bootstrap partition is highly similar to others

    part_alt = {}
    df = {}
    B = {i:0 for i in range(num_bootstrap)}
    
    for i in range(0,num_bootstrap):
        G_alt = bootstrap_network_edgeweight(G)
        part_alt[i] = louvain_community_detection(G_alt, G_nn, implementation, Cg, nodelabel_map)
        df[i] = pd.DataFrame.from_dict(part_alt[i], orient='index').reset_index()
        
    # Figure out how much consensus there is between each partition
    pairs = list(itertools.combinations(range(0,num_bootstrap), 2))
    for i,j in pairs:
        
        # merge pair of dataframes so that they line up
        df3 = pd.merge(df[i], df[j], on='index', how='inner')
        
        df3 = df3[~df3["0_x"].isna()]
        df3 = df3[~df3["0_y"].isna()]

        # calculate mutual information on two partitions
        nmi = normalized_mutual_info_score(df3["0_x"], df3["0_y"])
        
        # this value measures how much consensus partition[i] has with all the others       
        B[i] = B[i] + nmi
 
    #finding the key to the largest value in the dictionary
    if num_bootstrap == 1:
        i= 0
    else:
        i = max(B.items(), key=operator.itemgetter(1))[0]
       
    return part_alt[i], df[i]

##################################################
def bootstrap_network_edgeweight(G):
# returns a network where the edge-weights are redrawn from 
# a Poisson distribution with mean = eij (original edge weight)
	
	Galt = nx.Graph()
    
    #draw a bootstrap edge weight +/- Normal(0, 0.05)
	edges = [(u,v, d['weight']+np.random.normal(loc = 0,scale = 0.05)) for u,v,d in G.edges(data=True)]
    
	Galt.add_weighted_edges_from(edges)
	
	return Galt


##################################################
# PART 2b: SPATIAL COMMUNITY STRUCTURE ADJUSTMENT
##################################################


##################################################
def increase_spatial_contiguity(G_nn, partition_orig, size_thresh, Q_thresh):
# contiguity = all parts of a district be physically connected; no land islands
# increases spatial cohesiveness by looking at each community and combining nodes with 
#   nearest neighbor communities if not connected to rest of community
# keeps going till new Qrel value is still within thresh of original Qrel value
# INPUTS: G is network, G_nn is nearest neighbor network, partition is a dataframe with node and moularity
#  classes and Q_thresh is a tolerance for the reduction in modularity (Q_relative)
    
    partition = partition_orig.copy()
    
    # get a list of all the unique modularity classes (except 'nan')
    unique_modclass = list(partition['modularity_class'].dropna().unique())
    
    # identify geographical components of each community
    components = []                  
    for c in unique_modclass:
        comm_members = partition.loc[partition['modularity_class']== float(c)]['node'].tolist() # get all nodes that have modularity_class ==c

        H = G_nn.subgraph(comm_members)
        
        # find small components (size_thresh counties or less) that are islands
        comps = [comp for comp in nx.connected_components(H) if len(comp) <= size_thresh]
        components.extend(comps)

       
    # try to make graph more contiguous starting from the smallest components       
    for comp in components:
        
        # get all nearest neighbors of nodes in this component
        nn = [list(G_nn.neighbors(node)) for node in comp] # is list of lists
        nearest_neigh= [node for l in nn for node in l if node not in comp] # collapse list of lists and remove those nodes that are in the component
        nearest_neigh = list(set(nearest_neigh)) # get unique nearest neighbors
                
        nearest_neigh_comm = partition.loc[partition.node.isin(nearest_neigh)]['modularity_class'].dropna().tolist()
        
        if nearest_neigh_comm:
            comm_common = Counter(nearest_neigh_comm).most_common()[0][0]
            #print(nearest_neigh_comm, Counter(nearest_neigh_comm).most_common(), Counter(nearest_neigh_comm).most_common()[0][0])
    
            # make all nodes in this component match most popular neighboring community
            partition.loc[partition.node.isin(comp), 'modularity_class'] = comm_common

    nmi = calc_mutual_information(partition_orig, partition)
    print("      NMI: ", nmi)
					
    return partition

##################################################
def fill_in_nans(G_nn, partition_orig, Q_thresh):
# contiguity = all parts of a district be physically connected; no land islands
# increases spatial cohesiveness by looking at each community and combining nodes with 
#   nearest neighbor communities if not connected to rest of community
# keeps going till new Qrel value is still within thresh of original Qrel value
# INPUTS: G is network, G_nn is nearest neighbor network, partition is a dataframe with node and moularity
#  classes and Q_thresh is a tolerance for the reduction in modularity (Q_relative)

    
    partition = partition_orig.copy()
            
    # add all the components that are nan
    components = []
    p = partition['modularity_class'].isna()
    comm_members = partition.node[p].tolist() # get all nodes that have modularity_class ==c
    H = G_nn.subgraph(comm_members) 
    components.extend(list(nx.connected_components(H)))

    # try to make graph more contiguous starting from the smallest components       
    for comp in components:
        
        # get all nearest neighbors of nodes in this component
        nn = [list(G_nn.neighbors(node)) for node in comp] # is list of lists
        nearest_neigh= [node for l in nn for node in l if node not in comp] # collapse list of lists and remove those nodes that are in the component
        nearest_neigh = list(set(nearest_neigh)) # get unique nearest neighbors
                
        nearest_neigh_comm = partition.loc[partition.node.isin(nearest_neigh)]['modularity_class'].dropna().tolist()
        
        if nearest_neigh_comm:
            comm_common = Counter(nearest_neigh_comm).most_common()[0][0]
    
            # make all nodes in this component match most popular neighboring community
            partition.loc[partition.node.isin(comp), 'modularity_class'] = comm_common
                 
    nmi = calc_mutual_information(partition_orig, partition)
    print("      NMI: ", nmi)
					
    return partition

    	
##################################################
# PART 3: ANALYZE COMMUNITY STRUCTURE
##################################################


#################################################################
def calc_modularity(network, part):
# calculates the Newman modularity and the relative modularity of the network
# inputs: the robust network, the louvain robust partition dictionary
# outputs: the relative q value and the q value
        
    # Calculate Newman Modularity
    Q = community.modularity(part, network, weight='weight')
    
    # Calculate relative modularity (Sah et al, 2018, PNAS)
    L = network.number_of_edges() # number of edges 
    unique_mod_classes = set(part.values()) # list of modularity classes
    
    Qmax = 0
    for mod in unique_mod_classes:
        nodes = [n for (n,p) in part.items() if p == mod]
        H = network.subgraph(nodes)
        L_k = H.number_of_edges()
        x = (L_k/L)
        Qmax = Qmax + x*(1-x)

    Qrel = Q/Qmax
    
    return Q, Qrel


##########################################################
def calc_similarity_silhouette(node, matrix, part):
# get similarity between a given node and its community (needed for silhouette score)
#inputs: a given node, the distance matrix, partition
# outputs: the similarity of that node to the rest of the nodes in the cluster
        
    # find community of node
    nodecomm = part.loc[part['node']==node]['modularity_class'].tolist()[0]
    
    # find other nodes in same community (and remove original node)
    other_nodes_in_comm = part.loc[part['modularity_class']== nodecomm]['node'].tolist()
    other_nodes_in_comm = [n for n in other_nodes_in_comm if n != node]

    # calc dist to each node in same community
    dists_node = matrix[node]
    dists_comm =  dists_node[other_nodes_in_comm].tolist() # row of distances from node to all other nodes in same comm
    
    # calc avg distance between node and all nodes in same community
    sim = np.mean(dists_comm)
    
    return sim

###########################################################
def calc_dissimilarity_silhouette(node, matrix, part):
# get dissimilarity between a given node and its community (needed for silhouette score)
#inputs: a given node, the distance matrix, partition
#outputs: the dissimilarity of that node to the rest of the nodes
    
    # find community of node
    nodecomm = part.loc[part['node']==node]['modularity_class'].tolist()[0]
        
    # find all other community ids
    #first get a list of all mod classes in partition, then remove the mod class of node from that
    mod_list = list(part['modularity_class'].unique())
    cleanedlist = [x for x in mod_list if str(x) != 'nan']
    other_mods = sorted(i for i in cleanedlist if i != nodecomm) #remove mod in question from all mod list 
      
    # create list of average distances to all other communities
    avg_dist = []
    for other_comm in other_mods:
            
        # find nodes in other community
        nodes_in_other_comm = part.loc[part['modularity_class'] == other_comm]['node'].tolist()
    
        # calc dist from node to all nodes in this community
        dists_node = matrix[node]
        dists_comm =  dists_node[nodes_in_other_comm].tolist() # list of distances from node to all other nodes in other comm
       
        # avg dist from node to this community
        avg_dist.append(np.mean(dists_comm))
        
   # print('         ', avg_dist)
        
    # dissimilarity = minimum average distance to any other community
    if avg_dist:
        dissim = min(avg_dist)
    else:
        dissim = float('NaN')        
    
    return dissim


###############################################################
def calc_silhouette(all_county_timeseries, part):
# caculate silhouette score of community partition
# inputs: dist matrix, partition dataframe
# output: mean silhouette score
    
    # calculate euclidean distance between county timeseries
    corr_df = calc_timeseries_correlations(all_county_timeseries) # calculates pearson's correlations
    dist_matrix_df = 1 - corr_df # distance = 1-correlation
    
    #print(corr_df.head())
    
    #print(dist_matrix.head())
    
    # get list of counties and node ids           
    list_counties = list(part['node'].unique())
    #print(list_counties)
    
    # for each node (i.e. county fips), calculate silhouette score
    sil_dict = {}
    for node in list_counties:
    
        # calc similarity (a)
        a = calc_similarity_silhouette(node, dist_matrix_df, part)

        # calc dissimilarity (b)
        b = calc_dissimilarity_silhouette(node, dist_matrix_df, part)

        # calc silhouette ((b-a)/max(a,b))
        s = (b-a)/max(a, b)
        
        #print(node, a,b,s)
        
        sil_dict[node] = s
    
    # remove all Nans from silhouette score list
    sil_list = sil_dict.values()
    sil_list = [x for x in sil_list if str(x) != 'nan']
    
    #get the mean silhouette score
    mean_sil = np.mean(sil_list)
      
    return mean_sil, sil_list, sil_dict

##############################################   
def calc_mutual_information( df, df2):
# calculates mutual information between two partitions
# input: two filenames which contain partitions (fips, modularity_class)
    
    # need to make sure both partitions are ordered the same way
    #  (using inner to eliminate any fips missing in either partition)
    #  (The merge will create two columns modularity_class_x and modularity_class_y)
    df3 = pd.merge(df, df2, on='node', how='inner')
    
    # drop rows with NaNs; convert mods to ints
    df3 = df3.dropna(how='any')
    df3["modularity_class_x"] = df3["modularity_class_x"].astype(int)
    df3["modularity_class_y"] = df3["modularity_class_y"].astype(int)

    # calc match between paritionss
    numlist = len(list(df3.modularity_class_x))
    if numlist:
        match = sum([1 for a,b in zip(df3.modularity_class_x, df3.modularity_class_y) if a==b])/float(numlist) # print matches
    else:
        match= 0    
    nmi = normalized_mutual_info_score(df3.modularity_class_x, df3.modularity_class_y)
            
    return nmi, match


##################################################
# PART 5: OUTPUTTING RESULTS
##################################################

##################################################
def relabel_clusters(partition):
# RELABEL CLUSTERS SO THAT COLORS MATCH MAIN RESULTS
    
    part = partition
    
    # rename clusters into something else temporarily
    part.loc[part.modularity_class == 1, 'modularity_class'] = 'old_1'
    part.loc[part.modularity_class == 2, 'modularity_class'] = 'old_2'
    
    # rename clusters
    part.loc[part.modularity_class == 'old_1', 'modularity_class'] = 2
    part.loc[part.modularity_class == 'old_2', 'modularity_class'] = 1
        
    partition = part
    
    return partition


##################################################
# function to plot timeseries by community
def plot_timeseries(df, partition, filename,  year_of_analysis,  num_clusters, ylim):
    
    unique_modclass = list(partition['modularity_class'].unique())
    unique_modclass = [int(m) for m in unique_modclass if str(m) != 'nan']
    unique_modclass.sort()
       
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c', '#d95f02', '#7570b3', '#e7298a']
    
    # Plot timeseries
    hfont = {'fontname':'Helvetica'}
    fig, axes = plt.subplots(2, int(np.ceil(num_clusters/2)), figsize=(20,20))
    axes = axes.ravel()
    
    # for separate figure with all the averages
    #fig, ax2 = plt.subplots(figsize=(10,10))

    for mod in unique_modclass:
            
        # find nodes in ters community and get timeseries for just those nodes
        nodes_in_cluster = list(partition[partition.modularity_class == mod]['node'])
        flu_timeseries_mod = df[nodes_in_cluster]
               
        # plot all county timeseries (transparent)
        flu_timeseries_mod.plot(legend=False, color = colors[mod], alpha = 0.05, linewidth = 0.5, zorder = 8, ax=axes[mod])
        
        # plot smoothed average (opaque) for community
        flu_timeseries_mod_avg = flu_timeseries_mod.mean(axis=1)
        flu_timeseries_mod_avg.plot(legend=False, color = colors[mod], alpha = 1, linewidth = 4, zorder = 15, ax=axes[mod]);

        # put all the averages on one separate plot
        #flu_timeseries_mod_avg.plot(legend=False, color = colors[mod], alpha = 1, linewidth = 4, zorder = 15, ax=ax2);

        #axes[mod].set_title('Community '+str(mod),fontsize = 20)
        #axes[mod].set_xlabel("Week", fontsize = 20, **hfont)
        axes[mod].set_ylabel("Indoor Activity Seasonality", fontsize = 20, **hfont)
        axes[mod].tick_params(axis='x', labelsize= 16)
        axes[mod].tick_params(axis='y', labelsize= 16)
        axes[mod].set_ylim(ylim);

        if len(year_of_analysis) == 1:
            axes[mod].set_xticks([1, 5,9,14,18,23,27,31,36,40,44,49])
            axes[mod].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        elif len(year_of_analysis) == 2:
            axes[mod].set_xticks([1, 13, 26, 40, 53, 66, 79, 92])
            axes[mod].set_xticklabels(['Jan\n2018', 'Apr', 'Jul', 'Oct', 'Jan\n2019', 'Apr', 'Jul', 'Oct'])

        plt.savefig("timeseries_plot_"+str(filename)+'.png')

    return

##################################################
# function to plot county-level map with counties colored by community/module
def make_module_map(part_df, filename):
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    part_df2 = part_df.copy()
    part_df2['node'] = part_df2.node.astype(str).str.zfill(5)


    part_df2 = part_df2[~part_df2.modularity_class.isna()]
    part_df2['modularity_class'] = part_df2.modularity_class.astype(int)
    part_df2['modularity_class'] = part_df2.modularity_class.replace(0, 'A')
    part_df2['modularity_class'] = part_df2.modularity_class.replace(1, 'B')
    part_df2['modularity_class'] = part_df2.modularity_class.replace(2, 'C')
    part_df2['modularity_class'] = part_df2.modularity_class.replace(3, 'D')

    fig = px.choropleth(part_df2,                 # name of your dataframe
                        geojson=counties,
                        locations='node', # name of column in df that has the county fips
                        color='modularity_class',      # name of column in df that has the data you want to plot
                        color_discrete_map={"A":'#a6cee3',"B":'#1f78b4',"C":'#b2df8a',"D":'#33a02c'},#px.colors.qualitative.Safe,
                        scope='usa',
                       )

    fig.update_traces(marker_line_width=0.1,  # controls county border line width
                      marker_opacity=0.85,  # changes fill color opacity to let state borders through
                      marker_line_color='#262626',  # controls county border color; needs to be darker than "states"
                      )
    fig.update_layout(coloraxis_showscale=False, showlegend=False)

    fig.show()
    
    fig.write_image("modulemap_"+str(filename)+".png", scale=10, engine = 'kaleido')

    
    return