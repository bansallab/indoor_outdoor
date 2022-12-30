#!/usr/bin/env python

# # Network-based timeseries clustering
# #### Approach: calculate correlations between county timeseries; represent correlations as a county network, with dropping low weight edges; and run community structure detection on the network to identify groups of nodes that are more similar to each out in terms of dynamics
# 
# #### Author: Shweta Bansal
# #### Started Date: July 12, 2021
# #### Updated: Dec 23, 2022


import os
import networkx as nx
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import random as rnd

# import functions for the time-series clustering method
import epi_geo_functions as fegf

# run hierarchical clustering
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import fcluster


###########################################
# CHANGE PARAMETERS HERE FOR REST OF ANALYSIS

year_of_analysis = [2018,2019] #baseline years only

years_label = "_".join([str(y) for y in year_of_analysis])

filename = years_label+'_Dec2022_final'

threshold = 4 # absolute threshold for mean centered indoor activity measure

corr_percentile = 90 # percentile for the minimum time series correlation between counties (90 = default)

num_bootstrap = 25 # number of bootstrap networks over which to do analysis

drop_small = 10 # drop small communities of at most this many counties

contiguity_threshold = 2 # merge any island communities with neighboring cluster

sandbox = False # make true if shweta debugging

rolling = True

num_weeks = 4 # number of weeks over which to do rolling mean



###########################################
# LOAD DATA & CLEAN

# Load indoor/outdoor data for all years, take rolling mean

df_sm = pd.read_csv("indoor_activity_data/indoor_activity_2018_2020.csv") 

# fix date format
df_sm['date'] = pd.to_datetime(df_sm.date, format='%Y-%m-%d')

# take rolling mean
# number of weeks to roll defined above
# smoothing doesn't affect clustering results, but makes time series plots cleaner
if rolling:
    ct_list = list(df_sm.county.unique())
    dfn = pd.DataFrame()
    for ct in ct_list:
        dfx = df_sm[df_sm.county==ct]

        # rolling average of time series
        dfx = dfx.sort_values(by='date')
        dfx = dfx[['date', 'indoor_activity']]
        dfx = dfx.set_index('date')
        dfx = dfx.rolling(num_weeks).mean()
        dfx = dfx.reset_index()
        dfx['county'] = ct

        dfn = pd.concat([dfn, dfx], ignore_index=True)
    df_sm = dfn.copy()
    
df_sm = df_sm[['date', 'county', 'indoor_activity']]

df_sm.tail()



###########################################
# PREPARE TIMESERIES DATA FOR CLUSTERING

df_time = df_sm.copy()

# only keep data for years of interest
df_time['year'] = df_time.date.dt.year
df_time = df_time[df_time.year.isin(year_of_analysis)]
df_time = df_time[['county', 'indoor_activity', 'date']]

# convert to long to wide format where rows = weeks, columns = fips
df_matrix = df_time.pivot(index = 'date', columns='county', values='indoor_activity').reset_index(drop=True)

# keep unnormalized copy of df_matrix
df_matrix_unnorm = df_matrix.copy()

# z-normalize time series
m = df_matrix.mean(axis=0) # take mean for each col (i.e. for each county)
s = df_matrix.std(axis=0) # take std for each col
df_matrix = df_matrix.sub(m, axis=1) # subtract county mean from county time series
df_matrix = df_matrix.div(s, axis=1) # divide county time series by county stdev

# clean up dataframe by making all Nans 0    
df_matrix = df_matrix.fillna(0)

df_matrix.head()



###########################################
# CREATE CORRELATION MATRIX

# get correlation matrix between timeseries
corr_df = fegf.calc_timeseries_correlations(df_matrix)
corr_df.head()



###########################################
# CREATE NETWORK
                                            
# create network from correlation matrix
threshold = np.percentile(corr_df.values.tolist()[0], corr_percentile)
network = fegf.create_network(corr_df, threshold, df_sm, False)
                                            
# create nearest neighbor network
G_nn = fegf.make_nn_network("./", False)



############################################
# PERFORM TIME SERIES CLUSTERING THROUGH LOUVAIN COMMUNITY STRUCTURE

# run community structure detection using Louvain algo
part, dftemp = fegf.comm_struct_robustness(network, G_nn, 'louvain_igraph', num_bootstrap)
part_df = fegf.reduce_num_communities(part, drop_small) # drop communities that are smaller than dropsmall of number of counties

# save to file
part_df.to_csv("community_structure_"+filename+'.csv', index=False)

num_clusters = len(part_df.modularity_class.unique())

# output number of clusters and sizes of each cluster
print(part_df.modularity_class.value_counts())



#############################################
# POST PROCESS RESULTS

# make community structure feasible
part_nonan_df = fegf.fill_in_nans(G_nn, part_df, 0.15)
part_contig_df = fegf.increase_spatial_contiguity(G_nn, part_nonan_df, contiguity_threshold, 0.15)

part_nonan_df.to_csv("community_structure_nonan_"+filename+'.csv', index=False)
part_contig_df.to_csv("community_structure_feasible_"+filename+'.csv', index=False)



#############################################
# OUTPUT RESULTS

# plot time series (non-zscored data)
fegf.plot_timeseries(df_matrix_unnorm, part_df, filename+'_non_zscore', year_of_analysis, num_clusters, [0.25,2])

# plot time series (z-normalized)
#fegf.plot_timeseries(df_matrix, part_df, filename+'_zscore',  year_of_analysis, num_clusters, [-3,3])

# make map
fegf.make_module_map(part_nonan_df,filename)

# make feasible map
fegf.make_module_map(part_contig_df,filename+'_feasible')



#############################################
# COMPARE RESULTS TO HIERARCHICAL CLUSTERING

import importlib
importlib.reload(fegf)

# prepare matrix
df_matrix_hc = df_matrix.fillna(0).transpose() # needs to be county x week
df_matrix_hc = df_matrix_hc.loc[~(df_matrix_hc==0).all(axis=1)] # remove rows with all 0s
list_nodes = list(df_matrix_hc.index)

part_hier_clust = {} # dictionary of dataframes
for num_clusters in range(2,5): # vary number of clusters in partition
    
    # set up the time series linkage matrix for clustering
    Z = hac.linkage(df_matrix_hc, method='ward', metric='euclidean')

    # do time series clustering
    results = fcluster(Z, t=num_clusters, criterion='maxclust')

    # the results just tell you which partition each node (animal) is in, so this attaches the node ids to the cluster ids
    partition = dict(zip(list_nodes, results))
    
    # reduce small communities
    part_hier_clust[num_clusters] = fegf.reduce_num_communities(partition, 5)
    
    # output module number and sizes
    print(part_hier_clust[num_clusters].modularity_class.value_counts())
    

#############################################
# For num_clusters = 3, output results and quantify similartity to network clustering partition
num_clusters = 3

# relabel partitions to match the ordering of the main results
part_hier_clust[num_clusters] = fegf.relabel_clusters(part_hier_clust[num_clusters])

# output partition
part_hier_clust[num_clusters].to_csv("community_structure_hierclust_"+filename+'.csv', index=False)

# plot time series (non-zscored data)
fegf.plot_timeseries(df_matrix_unnorm, part_hier_clust[num_clusters], filename+'_hierclust_'+str(num_clusters), year_of_analysis, num_clusters, [0.25,2])

# make map
fegf.make_module_map(part_hier_clust[num_clusters],filename+'_hierclust_'+str(num_clusters))

# calculate mutual information on two partitions
part_network = pd.read_csv('community_structure_[2018, 2019]_Dec2022_corr90.csv') # this is the 90th percentile correlation network's comm struct results
df3 = pd.merge(part_network, part_hier_clust[num_clusters], on='node', how='inner') # merge two partitions by node
nmi = fegf.normalized_mutual_info_score(df3["modularity_class_x"], df3["modularity_class_y"])
print(nmi)

numlist = len(list(df3.modularity_class_x))
match = sum([1 for a,b in zip(df3.modularity_class_x, df3.modularity_class_y) if a==b])/float(numlist)
print(match)






