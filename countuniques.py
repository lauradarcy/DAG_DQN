
from environments.Environment import environment
import numpy as np
env = environment(number_inputs=[0], node_types=1, max_number_nodes=7, loss=0)
env.reset()
correct_combos = 0
total_combos = 0




'''correct_combos, total_combos = check_list(correct_combos,total_combos)
print(correct_combos)
print(total_combos)
print(str((correct_combos/total_combos)*100.0)+'%')'''

def run():
    correct_combos = 0
    total_combos = 0
    correct_combos, total_combos = check_list(correct_combos, total_combos)
    print(correct_combos)
    print(total_combos)
    percent = correct_combos/total_combos*100
    print(percent, '%')
    #print('done')
    return percent

def check_list(correct_combos, total_combos):
    actions_list = env.eligible_actions.copy()
    for action in actions_list:
        env + action
        correct_combos, total_combos = check_list(correct_combos,total_combos)
        env - action

    if np.shape(env.state)[0]==np.shape(env.truth_state)[0]:
        #print('final:\n', env.state)
        if env.reward > 0:
            correct_combos += 1
        total_combos += 1
    return correct_combos,total_combos

percents = 0
'''for i in range(10):
    env = workflow_env(number_inputs=[0], node_types=2, max_number_nodes=6, loss=0)
    env.reset()

    percent_new=run(7,1)
    percents += percent_new
print('final',str(percents/10) + '%')'''

run()

def num_final_states(size,types):
    states = 1
    for i in range(1,size):
        states *= types*(2**i-1)
    print(states)

num_final_states(7,1)