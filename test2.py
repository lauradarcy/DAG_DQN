from agents.Agent_Type3 import RL3
# from agents.Agent_Type2 import RL2
from agents.Random_Agent import R_Agent
from agents.Standard_Agent import S_Agent
from agents.Accumulated_Agent import A_Agent
from agents.Hindsight_Agent import H_Agent
import time
from matplotlib import pyplot as plt
import random
import os
import argparse

import concurrent.futures

file_random_str = str(random.randint(1, 10000))


def mean(a):
    return sum(a) / len(a)


def run_thread(j):
    learning_agents = [(H_Agent(graph_size=size, node_types=types, environment_name=environment_type,
                                total_episodes=episodes) if agent_type == 'hindsight' else
                        R_Agent(graph_size=size, node_types=types, environment_name=environment_type,
                                total_episodes=episodes) if agent_type == 'random'
                        else A_Agent(graph_size=size,
                                     node_types=types,
                                     environment_name=environment_type) if agent_type == 'accumulated' else
                        S_Agent(graph_size=size, node_types=types, environment_name=environment_type,
                                total_episodes=episodes)) for
                       (size, types, agent_type, environment_type) in list_of_agent_values]
    agent_name, environment_name = str(list_of_agent_values[0][2]), str(list_of_agent_values[0][3])
    os.makedirs('./data/' + agent_name + '/' + environment_name, exist_ok=True)
    os.makedirs('./data/' + agent_name, exist_ok=True)

    for i in range(episodes):
        if i % (save_rate) == 0 and i > 0:
            run_data[int(i / save_rate) - 1].append(learning_agents[0].success_val.copy())
            if len(run_data[int(i / save_rate) - 1]) == runs:
                '''
                all threads have completed up to this episode, so store data and make a rudimentary plot
                '''
                datafilename = './data/' + agent_name + '/' + environment_name + '/data_' + file_random_str + '_' + \
                               str(runs) + 'r-' + str(i) + 'e.dat'
                with open(datafilename, 'w') as f:
                    f.write('runs: {}, episodes: {}'.format(runs, i) + '\n')
                    # for data in run_data[int(i/save_rate)-1]:
                    f.write(str(run_data[int(i / save_rate) - 1]) + '\n')
                print('{} episodes of {} complete for all {} runs'.format(i, episodes, runs))

                plots = []

                plots.append(list(map(mean, zip(*run_data[int(i / save_rate) - 1]))))
                colors = ['r', 'g', 'b', 'gold', 'c', 'mediumblue', 'darkmagenta', 'dimgray']
                x = list(range(len(plots[0])))

                fig = plt.figure()
                for n in range(len(plots)):
                    plt.plot(x, plots[n], color=colors[n], label=labels[n])

                plt.legend()
                plotfilename = './data/' + agent_name + '/' + environment_name + '/plot_' + file_random_str + '_' + \
                               str(runs) + 'r-' + str(i) + 'e.png'

                print(plotfilename)
                fig.savefig(plotfilename, dpi=fig.dpi)
        for index, agent in enumerate(learning_agents):
            agent.epsilon_greedy()
            agent.greedy()

    for i in range(len(learning_agents)):
        run_data_final[i].append(learning_agents[i].success_val)


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
    parser.add_argument("-s", "--save_rate", type=int,
                        help="number of episodes to save data at", default=40)
    args = parser.parse_args()

    runs = args.runs
    episodes = args.episodes
    save_rate = args.save_rate
    list_of_agent_values = [
        (int(args.list_of_agent_values[i]), int(args.list_of_agent_values[i + 1]), args.list_of_agent_values[i
                                                                                                             + 2],
         args.list_of_agent_values[i + 3]) for i in range(len(args.list_of_agent_values))
        if i % 4 == 0]

    '''--------------------------------------------
    |   set labels, print information at start    |
    -----------------------------------------------'''

    file_label = '_' + '_'.join(
        ['-'.join([str(value) for value in agent]) for agent in list_of_agent_values]) + '-' + str(runs) + '-'
    labels = ['{} node graph, {} node type(s), agent type {}, environment type {}'.format(graph_size, node_types,
                                                                                          agent_type, environment_type)
              for (graph_size, node_types, agent_type, environment_type)
              in list_of_agent_values]

    print('{} runs of {} episodes each'.format(runs, episodes))
    print(labels[0])

    '''--------------------------------------------
    |        begin data collection                |
    -----------------------------------------------'''
    run_data = [[] for i in range(int(episodes / save_rate))]
    run_data_final = [[] for i in list_of_agent_values]
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=runs) as executor:
        executor.map(run_thread, range(int(runs)))
    thread_time = time.time() - start_time

    '''--------------------------------------------
    |        store data                           |
    -----------------------------------------------'''

    filestr = file_label + file_random_str
    # datafilename = 'data/data' + filestr + '.dat'
    agent_name_file, environment_name_file = str(list_of_agent_values[0][2]), str(list_of_agent_values[0][3])
    datafilename = './data/' + agent_name_file + '/' + environment_name_file + '/FINALdata' + file_label + str(
        episodes) + \
                   '_' + file_random_str + '.dat'
    with open(datafilename, 'w') as f:
        f.write('runs: {}, episodes: {} ;'.format(runs, episodes) + ';'.join(labels) + 'thread time:' + str(
            thread_time) + '\n')
        for data in run_data_final:
            f.write(str(data) + '\n')

    '''--------------------------------------------
    |        make fast plot for data              |
    -----------------------------------------------'''

    plots = []

    for data in run_data_final:
        plots.append(list(map(mean, zip(*data))))

    colors = ['r', 'g', 'b', 'gold', 'c', 'mediumblue', 'darkmagenta', 'dimgray']

    x = list(range(len(plots[0])))

    fig = plt.figure()
    for i in range(len(plots)):
        plt.plot(x, plots[i], color=colors[i], label=labels[i])

    plt.legend()
    plotfilename = './data/' + agent_name_file + '/' + environment_name_file + '/FINALplot_' + str(runs) + 'r-' + str(
        episodes) + 'e_' + file_random_str + '.png'

    print(plotfilename)
    fig.savefig(plotfilename, dpi=fig.dpi)
    print('thread time: {}'.format(thread_time))
