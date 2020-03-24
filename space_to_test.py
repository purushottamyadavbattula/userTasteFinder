import pandas as pd
import json
import io
def best_node_to_start():
    df = pd.read_csv('graph.csv', index_col=0)
    number_of_ones = []
    for i in range(1, 101):
        number_of_ones.append((df.loc[i, :].to_list()).count(1))
    return number_of_ones
def graph_to_list():
    df=pd.read_csv('graph.csv',index_col=0)
    file_tmp=io.open('graph_list.csv','w')
    dict_temp={}
    for i in range(100):
        temp=df.iloc[i,:].to_list()
        temp=[index+1 for index,val in enumerate(temp) if val==1]
        #print(temp)
        dict_temp[i+1]=temp
        #break
    file_tmp.write(json.dumps(dict_temp))
    file_tmp.close()
#print(best_node_to_start())
graph_to_list()