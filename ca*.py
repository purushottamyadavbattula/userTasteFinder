# imports
import numpy as np
import io

# data set reader


def plot_them()


def read_dataset(dataset_name='dataset.csv'):
    dataset_ref = open(dataset_name, 'r')
    return dataset_ref.readlines()


#
def mahatten_cal(node_1, node_2, size=6):
    node_1 = list(map(float, node_1.split(',')))[1:]
    node_2 = list(map(float, node_2.split(',')))[1:]
    sum_here = 0
    for i in range(size):
        sum_here += abs(node_1[i]-node_2[i])
    # print(sum_here)
    return sum_here

# print(mahatten_cal(dataset_ref[3],dataset_ref[4]))


def create_graph_adj(dataset_ref, size=99, arr_size=6):
    intermediate_results = []
    adjency_matrix = np.zeros((size, size), dtype='int')
    # print(adjency_matrix.shape)
    for i in range(1, size):
        for j in range(i, size):
            temp = mahatten_cal(dataset_ref[i], dataset_ref[j], size=arr_size)
            intermediate_results.append(temp)
            if(temp <= mahatten_distance_cutoff):
                adjency_matrix[i][j] = 1
            else:
                adjency_matrix[i][j] = 0
    # return [adjency_matrix[1:], intermediate_results]
    return adjency_matrix


def dataset_sum_row(dataset_ref, size=100):
    sums = []
    # 1 for removing the column names in csv
    for i in range(1, size):
        sum_here = 0
        line_1 = i
        line_2 = i+1
        # print(lines[line_1], lines[line_2])
        lst = list(map(float, dataset_ref[line_1].split(',')))
        lst2 = list(map(float, dataset_ref[line_2].split(',')))
        # print(sum(lst[1:]))
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


# intilisations
dataset_ref = read_dataset(dataset_name='dataset.csv')
sum_dataset = dataset_sum_row(dataset_ref, size=100)
mean = dataset_mean(sum_dataset)
avg = dataset_avg(sum_dataset)

# to construct a graph which is sparse
mahatten_distance_cutoff = avg if avg < mean else mean
# print(mahatten_distance_cutoff)

# avilable tastes
tastes = ['item', 'sweet', 'salty', 'bitter', 'sour', 'umami', 'astringency']

closed_ele = []
open_ele = []
distance_between_ele = 1

# creates a connected adjency graph shape(100,100) accessed by [99][99]
adj_graph = create_graph_adj(dataset_ref=dataset_ref, arr_size=6, size=100)

for_plot = io.open('graph.csv', 'a', encoding='utf-8')
for_plot.write(','.join(str(i) for i in range(1, 101)))
for index, i in enumerate(adj_graph):
    for_plot.write('\n'+str(index)+","+','.join(str(j) for j in i))
for_plot.close()
