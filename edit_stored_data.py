from matplotlib import pyplot as plt
import numpy as np

import re
def mean(a):
    return sum(a) / len(a)

plt.style.use('seaborn-paper')


'''
data/standard/standard/FINALdata_4-1-standard-standard-20-10000_5519.dat
data/standard/standard/FINALdata_5-1-standard-standard-20-10000_7621.dat
data/standard/standard/FINALdata_5-2-standard-standard-20-10000_3826.dat
data/standard/standard/FINALdata_5-3-standard-standard-20-10000_4989.dat
data/standard/standard/FINALdata_6-1-standard-standard-20-10000_2440.dat
data/standard/standard/FINALdata_8-1-standard-standard-20-10000_9079.dat


data/hindsight/standard/FINALdata_4-1-hindsight-standard-20-10000_644.dat
data/hindsight/standard/FINALdata_5-1-hindsight-standard-20-10000_2914.dat
data/hindsight/standard/FINALdata_5-2-hindsight-standard-20-10000_6580.dat
data/hindsight/standard/FINALdata_5-3-hindsight-standard-20-10000_2226.dat
data/hindsight/standard/FINALdata_6-1-hindsight-standard-20-10000_6529.dat

new data style:
'data/hindsight/workflow/FINALdata_4-1-hindsight-workflow-20-10000_5463.dat',
'data/hindsight/workflow/FINALdata_5-1-hindsight-workflow-20-10000_7379.dat',
'data/hindsight/workflow/FINALdata_5-2-hindsight-workflow-20-10000_3261.dat',

'data/random/standard/FINALdata_4-1-random-standard-20-10000_5145.dat',
'data/random/standard/FINALdata_5-1-random-standard-20-10000_613.dat',
'data/random/standard/FINALdata_5-2-random-standard-20-10000_3372.dat',
'data/random/standard/FINALdata_5-3-random-standard-20-10000_7925.dat',

'data/random/workflow/FINALdata_4-1-random-workflow-20-10000_9797.dat',
'data/random/workflow/FINALdata_5-1-random-workflow-20-10000_7064.dat',
'data/random/workflow/FINALdata_5-2-random-workflow-20-10000_2423.dat',
'data/random/workflow/FINALdata_5-3-random-workflow-20-10000_355.dat',
'data/random/workflow/FINALdata_6-1-random-workflow-20-10000_3590.dat',
'data/random/workflow/FINALdata_8-1-random-workflow-20-10000_7589.dat'
'''
filenames = []
datafilename = 'data/standard/single/FINALdata_4-1-standard-single-20-10000_3280.dat'
with open(datafilename) as f:
    data_list = f.readlines()
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#print(data_list[1])

def get_useable_data(string_data):
    p = re.compile('\[([^\[\]]+)')
    matches = re.findall(p, string_data)
    list_useable_data = [[float(i) for i in run.split(',')] for run in matches]
    return list_useable_data


run_data = [get_useable_data(data_list[i]) for i in range(1,len(data_list))]
#print(len(run_data[-2]))

#run_data_old = run_data[-2]

for n,i in enumerate(run_data):
    #print(n,len(i),len(i[0]), i[0][0])
    print(n, len(i))

plots = [list(map(mean, zip(*run_data_item))) for run_data_item in run_data]
#run_data1 = get_useable_data(data_list[1])
#plots = [list(map(mean, zip(*run_data_old)))]
print('avg',sum(plots[0]),len(plots[0]))


#plots = [avgs1]
colors = ['r', 'g', 'b']
labels = ['one node type', 'two node types', 'four node types']

#print(avgs1[:5])

y = [running_mean(plots[0],3000)]
x = list(range(len(y[0])))
for i in range(len(y)):
    plt.plot(x, y[i], label=labels[i])
plt.xlabel('episode')
plt.ylabel('success rate')
plt.title(r'success rate of greedy policy over time')

#sns.lineplot(data=data, palette="tab10", linewidth=2.5)
plt.legend()
plt.show()