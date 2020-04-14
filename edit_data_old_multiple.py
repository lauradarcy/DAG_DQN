from matplotlib import pyplot as plt
import numpy as np

import re
def mean(a):
    return sum(a) / len(a)

def mean2(a):
    return running_mean_fast(mean(a),100)
plt.style.use('seaborn-paper')


'''
'data/standard/standard/FINALdata_4-1-standard-standard-20-10000_5519.dat',
'data/standard/standard/FINALdata_5-1-standard-standard-20-10000_7621.dat',
'data/standard/standard/FINALdata_5-2-standard-standard-20-10000_3826.dat',
'data/standard/standard/FINALdata_5-3-standard-standard-20-10000_4989.dat',
'data/standard/standard/FINALdata_6-1-standard-standard-20-10000_2440.dat',
'data/standard/standard/FINALdata_8-1-standard-standard-20-10000_9079.dat',
'data/hindsight/standard/FINALdata_8-1-hindsight-standard-20-10000_946.dat',


'data/hindsight/standard/FINALdata_4-1-hindsight-standard-20-10000_644.dat',

'data/hindsight/standard/FINALdata_5-2-hindsight-standard-20-10000_6580.dat',
'data/hindsight/standard/FINALdata_5-3-hindsight-standard-20-10000_2226.dat',
'data/hindsight/standard/FINALdata_6-1-hindsight-standard-20-10000_6529.dat'

new data style:
data/hindsight/workflow/FINALdata_4-1-hindsight-workflow-20-10000_5463.dat
data/hindsight/standard/FINALdata_5-1-hindsight-standard-20-10000_2914.dat
data/hindsight/workflow/FINALdata_5-1-hindsight-workflow-20-10000_7379.dat
'''

'''list_of_files = ['data/hindsight/standard/FINALdata_4-1-hindsight-standard-20-10000_644.dat',
'data/standard/standard/FINALdata_4-1-standard-standard-20-10000_5519.dat',

'data/hindsight/standard/FINALdata_5-2-hindsight-standard-20-10000_6580.dat',
'data/standard/standard/FINALdata_5-2-standard-standard-20-10000_3826.dat',

'data/hindsight/standard/FINALdata_6-1-hindsight-standard-20-10000_6529.dat',
'data/standard/standard/FINALdata_6-1-standard-standard-20-10000_2440.dat',]'''

list_of_files = [
'data/standard/standard/FINALdata_4-1-standard-standard-20-10000_5519.dat',
]

data_list = []
for datafilename in list_of_files:
    with open(datafilename) as f:
        data_list.append(f.readlines())


#print(data_list[1])
def running_mean_fast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_useable_data(string_data):
    p = re.compile('\[([^\[\]]+)')
    matches = re.findall(p, string_data)
    list_useable_data = [[float(i) for i in run.split(',')] for run in matches]
    return list_useable_data

run_data_multiple = []
for data in data_list:
    run_data_multiple.append([get_useable_data(data[i]) for i in range(1,len(data))])

print(len(run_data_multiple[0][-2]))

run_data_old = [run_data[-2] for run_data in run_data_multiple]

'''
for new data:
'''
datafilename_new = 'data/standard/single/FINALdata_4-1-standard-single-20-10000_3280.dat'
with open(datafilename_new) as f:
    data_list_new = f.readlines()

run_data_new = [get_useable_data(data_list_new[i]) for i in range(1, len(data_list_new))]




for item in run_data_old:
    for n,i in enumerate(item):
        #print(n,len(i),len(i[0]), i[0][0])
        #end_of_list =
        i.extend(i[-100:])
        print(n, len(i), len(i[-100:]))

#plots = [list(map(mean, zip(*run_data_item))) for run_data_item in run_data]
#run_data1 = get_useable_data(data_list[1])
plots = [list(map(mean, zip(*run_data))) for run_data in run_data_old]
plots.extend([list(map(mean, zip(*run_data_item))) for run_data_item in run_data_new])
plots_running = []
for plot in plots:
    plots_running.append(running_mean(plot,50))
    print('avg',sum(plot),len(plot))


#plots = [avgs1]
colors = ['r', 'g', 'b']
'''labels = ['four node graph, DQN+PER', 'four node graph, DQN', 'five node graph with two node types, DQN+PER',
          'five node graph with two node types, DQN', 'six node graph, DQN+PER', 'six node graph, DQN', 'single']
'''
labels = ['binary action selection', 'single action selection']
#print(avgs1[:5])
x = list(range(len(plots_running[0])))

for i in range(len(plots)):
    plt.plot(x, plots_running[i], label=labels[i])
plt.xlabel('number of episodes')
plt.ylabel('average success rate')
plt.title(r'success rate of greedy policy over time')

#sns.lineplot(data=data, palette="tab10", linewidth=2.5)
plt.legend()
plt.show()