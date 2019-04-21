from agents.Agent_Type3 import RL3
#from agents.Agent_Type2 import RL2
from agents.Random_Agent import R_Agent
from agents.Standard_Agent import S_Agent
from agents.Accumulated_Agent import A_Agent
from agents.Hindsight_Agent import H_Agent
import time
from matplotlib import pyplot as plt
import random

import argparse

import concurrent.futures

def run_thread(j):

    learning_agents = [(H_Agent(graph_size=size, node_types=types, environment_name=environment_type) if agent_type ==
                        'hindsight' else R_Agent(graph_size=size, node_types=types, environment_name=environment_type) if agent_type ==
                        'random' else A_Agent(graph_size=size, node_types=types, environment_name=environment_type) if agent_type ==
                        'accumulated' else S_Agent(graph_size=size, node_types=types,environment_name=environment_type)) for
                       (size, types, agent_type, environment_type) in list_of_agent_values]

    for i in range(episodes):
        if i % (500) == 0:
            print('t',j + i / episodes, '/', runs)
        for index, agent in enumerate(learning_agents):
            agent.epsilon_greedy()
            agent.greedy()

    for i in range(len(learning_agents)):
        run_data[i].append(learning_agents[i].success_val)


if __name__ == "__main__":

    '''-------------------------
    |        parse arguments    |
    ----------------------------'''

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list_of_agent_values", type=str, nargs='*',
                        help="list of env traits, tuple (graph_size,node_types,sample_size,loss,agent_type)",
                        default=[4, 1, 'hindsight', 'random'])
    parser.add_argument("-r", "--runs", type=int,
                        help="number of workflow_env runs", default=1)
    parser.add_argument("-e", "--episodes", type=int,
                        help="number of episodes per learning run", default=40)
    args = parser.parse_args()

    runs = args.runs
    episodes = args.episodes
    list_of_agent_values = [(int(args.list_of_agent_values[i]), int(args.list_of_agent_values[i + 1]), args.list_of_agent_values[i
                            + 2], args.list_of_agent_values[i + 3]) for i in range(len(args.list_of_agent_values))
                            if i % 4 == 0]

    '''--------------------------------------------
    |   set labels, print information at start    |
    -----------------------------------------------'''

    file_label = '_' + '_'.join(['-'.join([str(value) for value in agent]) for agent in list_of_agent_values]) + '-' + str(runs) + '-' + str(episodes) + '_'
    labels = ['{} node graph, {} node type(s), agent type {}, workflow_env type {}'.format(graph_size, node_types,
                agent_type, environment_type) for (graph_size, node_types, agent_type, environment_type)
                in list_of_agent_values]

    print('{} runs of {} episodes each'.format(runs, episodes))
    print(labels[0])

    '''--------------------------------------------
    |        begin data collection                |
    -----------------------------------------------'''
    run_data = [[] for i in list_of_agent_values]
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(run_thread, range(int(runs)))
    thread_time = time.time()-start_time

    '''--------------------------------------------
    |        store data                           |
    -----------------------------------------------'''

    filestr = file_label + str(random.randint(1, 10000))
    datafilename = 'data/data' + filestr + '.dat'
    with open(datafilename, 'w') as f:
        f.write('runs: {}, episodes: {}'.format(runs,episodes)+';'.join(labels) + 'thread time:' + str(thread_time) +'\n')
        for data in run_data:
            f.write(str(data) + '\n')

    '''--------------------------------------------
    |        make fast plot for data              |
    -----------------------------------------------'''

    plots = []

    def mean(a):
        return sum(a) / len(a)

    for data in run_data:
        plots.append(list(map(mean, zip(*data))))

    colors = ['r', 'g', 'b', 'gold', 'c', 'mediumblue', 'darkmagenta', 'dimgray']

    x = list(range(len(plots[0])))

    fig = plt.figure()
    for i in range(len(plots)):
        plt.plot(x, plots[i], color=colors[i], label=labels[i])

    plt.legend()

    filename = 'plots/plot' + filestr + '.png'
    print(filename)
    fig.savefig(filename, dpi=fig.dpi)
    print('thread time: {}'.format(thread_time))
