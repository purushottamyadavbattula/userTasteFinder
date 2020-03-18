import io
import pandas as pd
import networkx as nkx
import matplotlib
import matplotlib.pyplot as plt
dataset = open('dataset.csv', 'r')
# for i in range(100):
#     print(dataset.readline())
lines = dataset.readlines()
sums = []
for i in range(1, 99):
    sum = 0
    line_1 = i
    line_2 = i+1
    #print(lines[line_1], lines[line_2])
    lst = list(map(float, lines[line_1].split(',')))
    lst2 = list(map(float, lines[line_2].split(',')))
    # print(sum(lst[1:]))
    sum = 0
    for i in range(1, 6):
        sum += abs(lst[i]-lst2[i])
    sums.append(sum)
# print(sums)
#print(max(sums), min(sums))
print('avg', (max(sums)+min(sums))/2)
sum = 0
for i in sums:
    sum += i
print("mean", sum/len(sums))
dataset.close()
input_data = pd.read_csv('graph.csv', index_col=0)
# print(input_data)
G = nkx.DiGraph(input_data.values)
nkx.draw(G, node_size=10, width=0.1, arrows=False)
# plt.savefig("Plotting_samples.pdf")
plt.show()
