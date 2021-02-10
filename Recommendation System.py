# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:11:36 2020

@author: knith
"""

print ()

import networkx
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx


# Read the data from amazon-books.csv into amazonBooks dataframe;
amazonBooks = pd.read_csv('D:/ASU/Sem 2/CIS 509 Data Mining 2/Homework/Assignment 3/SNACode/NetworkAnalysis/amazon-books.csv', index_col=0)

# Read the data from amazon-books-copurchase.adjlist;
# assign it to copurchaseGraph weighted Graph;
# node = ASIN, edge= copurchase, edge weight = category similarity
fhr=open("amazon-books-copurchase.edgelist", 'rb')
copurchaseGraph=networkx.read_weighted_edgelist(fhr)
fhr.close()
df2 = amazonBooks.copy()
df2.index.names = ['ASIN']


# Now let's assume a person is considering buying the following book;
# what else can we recommend to them based on copurchase behavior 
# we've seen from other users?
print ("Looking for Recommendations for Customer Purchasing this Book:")
print ("--------------------------------------------------------------")
purchasedAsin = '0805047905'
AvgRat = amazonBooks.loc[purchasedAsin,'AvgRating']

# Let's first get some metadata associated with this book
print ("ASIN = ", purchasedAsin) 
print ("Title = ", amazonBooks.loc[purchasedAsin,'Title'])
print ("SalesRank = ", amazonBooks.loc[purchasedAsin,'SalesRank'])
print ("TotalReviews = ", amazonBooks.loc[purchasedAsin,'TotalReviews'])
print ("AvgRating = ", AvgRat)
#print ("AvgRating = ", amazonBooks.loc[purchasedAsin,'AvgRating'])
print ("DegreeCentrality = ", amazonBooks.loc[purchasedAsin,'DegreeCentrality'])
print ("ClusteringCoeff = ", amazonBooks.loc[purchasedAsin,'ClusteringCoeff'])


# Now let's look at the ego network associated with purchasedAsin in the
# copurchaseGraph - which is esentially comprised of all the books 
# that have been copurchased with this book in the past
#     Get the depth-1 ego network of purchasedAsin from copurchaseGraph,
#     and assign the resulting graph to purchasedAsinEgoGraph.
# (1) YOUR CODE HERE: 
G = copurchaseGraph
n = purchasedAsin
ego = networkx.ego_graph(G, n, radius=1)
egoneighbors = [i for i in ego.neighbors(n)] 
print (egoneighbors)
purchasedAsinEgoGraph = ego


# Next, recall that the edge weights in the copurchaseGraph is a measure of
# the similarity between the books connected by the edge. So we can use the 
# island method to only retain those books that are highly simialr to the 
# purchasedAsin
# (2) YOUR CODE HERE: 
#     Use the island method on purchasedAsinEgoGraph to only retain edges with 
#     threshold >= 0.5, and assign resulting graph to purchasedAsinEgoTrimGraph
threshold = 0.5
purchasedAsinEgoTrimGraph = networkx.Graph()
for f, t, e in purchasedAsinEgoGraph.edges(data=True):
    if e['weight'] > threshold:
        purchasedAsinEgoTrimGraph.add_edge(f,t,weight=e['weight'])
        
pos=networkx.spring_layout(purchasedAsinEgoTrimGraph)
plt.figure(figsize=(10,10))
networkx.draw_networkx_nodes(purchasedAsinEgoTrimGraph,pos,node_size=1500)
networkx.draw_networkx_labels(purchasedAsinEgoTrimGraph,pos,font_size=20)
edgewidth = [ d['weight'] for (u,v,d) in purchasedAsinEgoTrimGraph.edges(data=True)]
networkx.draw_networkx_edges(purchasedAsinEgoTrimGraph,pos,width=edgewidth)
edgelabel = networkx.get_edge_attributes(purchasedAsinEgoTrimGraph,'weight')
networkx.draw_networkx_edge_labels(purchasedAsinEgoTrimGraph,pos,edge_labels=edgelabel,font_size=20)
plt.axis('off') 
plt.show()


# Next, recall that given the purchasedAsinEgoTrimGraph you constructed above, 
# you can get at the list of nodes connected to the purchasedAsin by a single 
# hop (called the neighbors of the purchasedAsin) 
# (3) YOUR CODE HERE: 
#     Find the list of neighbors of the purchasedAsin in the 
#     purchasedAsinEgoTrimGraph, and assign it to purchasedAsinNeighbors
purchasedAsinNeighbors = purchasedAsinEgoTrimGraph.neighbors(n)
df1 = pd.DataFrame(data =list(purchasedAsinNeighbors), columns = ['ASIN'])
df = pd.merge(df1, df2, on = 'ASIN')


# Next, let's pick the Top Five book recommendations from among the 
# purchasedAsinNeighbors based on one or more of the following data of the 
# neighboring nodes: SalesRank, AvgRating, TotalReviews, DegreeCentrality, 
# and ClusteringCoeff
# (4) YOUR CODE HERE: 
#     Note that, given an asin, you can get at the metadata associated with  
#     it using amazonBooks (similar to lines 29-36 above).
#     Now, come up with a composite measure to make Top Five book 
#     recommendations based on one or more of the following metrics associated 
#     with nodes in purchasedAsinNeighbors: SalesRank, AvgRating, 
#     TotalReviews, DegreeCentrality, and ClusteringCoeff. Feel free to compute
#     and include other measures if you like.
#     YOU MUST come up with a composite measure.
#     DO NOT simply make recommendations based on sorting!!!
#     Also, remember to transform the data appropriately using 
#     sklearn preprocessing so the composite measure isn't overwhelmed 
#     by measures which are on a higher scale.

df['ARFactor'] = df['AvgRating'].apply(lambda x: 0.5 if x<AvgRat else 1)

mean_srf = df['SalesRank'].mean()
max_srf = df['SalesRank'].max()
min_srf = df['SalesRank'].min()
df['SRFactor'] = df['SalesRank'].apply(lambda x:1-((((x-mean_srf)/(max_srf-min_srf))+1)/2))

max_dc = df['DegreeCentrality'].max()
df['DC'] = df['DegreeCentrality'].apply(lambda x:x/max_dc)
df['DCFactor'] = df['DC']*df['ClusteringCoeff']

df['Recommendation'] = df['ARFactor']*df['SRFactor']*df['DCFactor']

RecTable = df.sort_values(by='Recommendation', ascending=False)


# Print Top 5 recommendations (ASIN, and associated Title, Sales Rank, 
# TotalReviews, AvgRating, DegreeCentrality, ClusteringCoeff)
# (5) YOUR CODE HERE:
RecTable.drop(['Id', 'Categories', 'ARFactor', 'SRFactor', 'DC', 'DCFactor'], axis = 1, inplace = True)
print(RecTable[0:5])