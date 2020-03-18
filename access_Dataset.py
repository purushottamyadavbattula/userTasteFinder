import io
dataset = open('dataset.csv', 'r')
# for i in range(100):
#     print(dataset.readline())
lines = dataset.readlines()
print(lines[3])
lst = map(float, lines[3].split(','))
print(sum(lst))

dataset.close()
