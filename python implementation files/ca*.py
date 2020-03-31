# imports
import numpy as np
import io
import pandas as pd
import networkx as nkx
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from datetime import datetime
import json


# avilable tastes
tastes = ['item', 'sweet', 'salty', 'bitter', 'sour', 'umami', 'astringency']



def plot_to_file(adj_graph):
    if(os.stat('../csv_files/graph.csv').st_size==0):
        for_plot = io.open('../csv_files/graph.csv', 'a', encoding='utf-8')
    else:
        for_plot = io.open('../csv_files/graph.csv', 'w', encoding='utf-8').close()
        for_plot = io.open('../csv_files/graph.csv', 'a', encoding='utf-8')
    for_plot.write(','.join(str(i) for i in range(1, 101)))
    for index, i in enumerate(adj_graph):
        for_plot.write('\n'+str(index+1)+","+','.join(str(j) for j in i))
    for_plot.close()
    # return adj_graph


def plot_show():

    input_data = pd.read_csv('../csv_files/graph.csv', index_col=0)

    G = nkx.DiGraph(input_data.values)
    pos = nkx.spring_layout(G, k=0.15, iterations=20)
    nkx.draw(G, pos, node_size=10, width=0.1, arrows=False)
    # plt.savefig("Plotting_samples.pdf")
    plt.show()


# data set reader
def read_dataset(dataset_name='../csv_files/dataset.csv'):
    dataset_ref = open(dataset_name, 'r')
    return dataset_ref.readlines()


# calculates mahatten distance between two vectors
def mahatten_cal(node_1, node_2, size=6):
    node_1 = list(map(float, node_1.split(',')))[1:]
    node_2 = list(map(float, node_2.split(',')))[1:]
    sum_here = 0
    for i in range(size):
        sum_here += abs(node_1[i]-node_2[i])

    return sum_here




def create_graph_adj(dataset_ref, size=99, arr_size=6):
    intermediate_results = []
    adjency_matrix = np.zeros((size, size), dtype='int')

    for i in range(0, size):
        for j in range(i, size):
            temp = mahatten_cal(dataset_ref[i], dataset_ref[j], size=arr_size)
            intermediate_results.append(temp)
            if(temp <= mahatten_distance_cutoff):
                adjency_matrix[i][j] = 1
                adjency_matrix[j][i] = 1
            else:
                adjency_matrix[i][j] = 0

    return adjency_matrix


def dataset_sum_row(dataset_ref, size=100):
    sums = []
    # 1 for removing the column names in csv
    for i in range(1, size):
        sum_here = 0
        line_1 = i
        line_2 = i+1

        lst = list(map(float, dataset_ref[line_1].split(',')))
        lst2 = list(map(float, dataset_ref[line_2].split(',')))

        sum_here = 0
        for i in range(1, 6):
            sum_here += abs(lst[i]-lst2[i])
        sums.append(sum_here)
    return sums


def dataset_mean(sums):
    sum_here = 0
    for i in sums:
        sum_here += i
    return sum_here/len(sums)


def dataset_avg(sums):
    return (max(sums)+min(sums))/2


# for user orders
def mean_all(dataset_name='../csv_files/userOrders.csv'):
    user_mean_cluster = []
    data_frame = pd.read_csv(dataset_name)
    for taste in tastes:
        temp = data_frame[taste].to_list()
        user_mean_cluster.append((math.fsum(temp))/len(temp))
    return user_mean_cluster

# returning no of ones


def best_node_to_start(adj_graph):
    df = pd.read_csv('../csv_files/graph.csv', index_col=0)
    number_of_ones = []
    for i in range(1, 101):
        number_of_ones.append((df.loc[i, :].to_list()).count(1))
    return number_of_ones

# adj matrix to lst converter
def graph_to_list():
    df=pd.read_csv('../csv_files/graph.csv',index_col=0)
    file_tmp=io.open('../csv_files/graph_list.csv','w')
    dict_temp={}
    for i in range(100):
        temp=df.iloc[i,:].to_list()
        temp=[index+1 for index,val in enumerate(temp) if val==1]
        #print(temp)
        dict_temp[i+1]=temp
        #break
    file_tmp.write(json.dumps(dict_temp))
    file_tmp.close()
    return dict_temp


# basic bfs with limit (cunstomized version)

def bfs(graph, v,limit=10):
    count=0
    all = []
    Q = []
    Q.append(v)
    while Q != []:
        v = Q.pop(0)
        all.append(v)
        count+=1
        if(count>=limit):
            return all
        for n in graph[v]:
            if n not in Q and n not in all:
                Q.append(n)
    return all


def node_dist_cal(user_mean_sum, dataset_ref, indexes):
    sums_index = []
    for i in indexes:
        sum = 0

        temp = list(map(float, dataset_ref[i].split(',')))
        for j in range(7):
            sum += abs(temp[j]-user_mean_sum[j])
        sums_index.append(sum)

    return sums_index


# intilisations.iloc([0])
dataset_ref = read_dataset(dataset_name='../csv_files/dataset.csv')
sum_dataset = dataset_sum_row(dataset_ref, size=100)
mean = dataset_mean(sum_dataset)
avg = dataset_avg(sum_dataset)

# to construct a graph which is sparse
mahatten_distance_cutoff = avg if avg < mean else mean

# creates a connected adjency graph shape(100,100) accessed by [99][99]
adj_graph = create_graph_adj(dataset_ref=dataset_ref[1:], arr_size=6, size=100)
for i in range(0, 100):
    adj_graph[i][i] = 0

user_orders_ref = read_dataset(dataset_name='../csv_files/userOrders.csv')  # plot_show()

plot_to_file(adj_graph)
plot_show()

#gets avg for all items to iterate in the graph
user_mean_sum = mean_all(dataset_name='../csv_files/userOrders.csv')

#returns best node haing maximum connections
node_num = best_node_to_start(adj_graph)


node_ones_count = best_node_to_start(adj_graph)
node_to_start = node_ones_count.index(max(node_ones_count))

graph_lst=graph_to_list()
index_to_find = graph_lst[node_to_start]


#calculates distance from all nodes connected to best node and picks one
temp = node_dist_cal(user_mean_sum, dataset_ref, index_to_find)
node_to_start = index_to_find[temp.index(max(temp))]




# predition
date_time_str= str(datetime.now())
user_taste_matched_items = bfs(graph_to_list(), node_to_start,limit=20)
file_to_show_user = io.open('../csv_files/user menu.csv', 'a', encoding='utf-8')
file_to_show_user.write("\n"+date_time_str+"\n")
file_to_show_user.write('item,sweet,salty,bitter,sour,umami,astringency\n')
for i in user_taste_matched_items:
    file_to_show_user.write(dataset_ref[i])
file_to_show_user.close()
