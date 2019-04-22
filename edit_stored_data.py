from matplotlib import pyplot as plt

import re
def mean(a):
    return sum(a) / len(a)

plt.style.use('seaborn-paper')

datafilename = 'data/standard/standard/FINALdata_4-1-standard-standard-5-400_851.dat'
with open(datafilename) as f:
    data_list = f.readlines()

#print(data_list[1])

def get_useable_data(string_data):
    p = re.compile('\[([^\[\]]+)')
    matches = re.findall(p, string_data)
    list_useable_data = [[float(i) for i in run.split(',')] for run in matches]
    return list_useable_data

run_data = [get_useable_data(data_list[i]) for i in range(1,len(data_list))]

for n,i in enumerate(run_data):
    print(n,len(i), len(i[0]), i[0][0])
plots = [list(map(mean, zip(*run_data_item))) for run_data_item in run_data]
#run_data1 = get_useable_data(data_list[1])
#avgs1 = list(map(mean, zip(*run_data1)))
print('avg',sum(plots[0]))

#plots = [avgs1]
colors = ['r', 'g', 'b']
labels = ['one node type', 'two node types', 'four node types']

#print(avgs1[:5])
x = list(range(len(plots[0])))

for i in range(len(plots)):
    plt.plot(x, plots[i], label=labels[i])
plt.xlabel('episode')
plt.ylabel('success rate')
plt.title(r'success rate of greedy policy over time')

#sns.lineplot(data=data, palette="tab10", linewidth=2.5)
plt.legend()
plt.show()