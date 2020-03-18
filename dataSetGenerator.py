import random
import io
dataset = io.open('dataset.csv', 'a', encoding='utf-8')
dataset.write('item,sweet,salty,bitter,sour,umami,astringency')
for iter_var in range(0, 100):
    row = [random.random() for i in range(6)]
    row_sum = sum(row)
    row_final = [i/row_sum for i in row]
    temp = ','.join(str(item) for item in row_final)
    dataset.write("\n"+str(iter_var)+","+temp)
dataset.close()
