import os
import random
import concurrent.futures
agent_type = 'hi'

os.makedirs('./data/hello',exist_ok=True)

runs = 20

episodes = 50

run_data = [[] for i in range(int(episodes/5))]
print(len(run_data))

values = []
def run_thread(j):
    os.makedirs('./data/'+agent_name, exist_ok=True)
    for i in range(episodes):
        if i % 5 == 0:
            print(len(values),int(i/5))
            run_data[int(i/5)].append(values.copy())
            if len(run_data[int(i/5)])==20:
                datafilename = './data/' + str(agent_name) + '/' + str(int(i/5)) + '.dat'
                with open(datafilename, 'w') as f:
                    f.write('runs: {}, episodes: {}'.format(runs, i) + '\n')
                    for data in run_data[int(i/5)]:
                        f.write(str(data) + '\n')
        values.append(random.randint(1, 50))

for i in run_data:
    for a in i:
        print(len(i),len(a),a)
agent_name = 'testlolol5'

with concurrent.futures.ThreadPoolExecutor(max_workers=runs) as executor:
    executor.map(run_thread, range(int(runs)))

