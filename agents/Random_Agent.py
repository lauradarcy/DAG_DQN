from agents.network.DQN import DQN

from environments.Environment import environment
from environments.WorkflowEnvironment import workflow_env
import torch
import random

from agents.Replay_v2 import ExperienceReplay
import torch.optim as optim


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''make it so buffer stores time of memory, sort by unusual and then time, also store best action/best actions state'''

class R_Agent:
    def __init__(self, input_node_type=[0], graph_size=4, node_types=1, environment_name=None, total_episodes=None):
        #print('random agent')
        self.max_number_nodes = graph_size
        self.node_types = node_types
        self.initial_nodes = input_node_type

        self.episode_length = 30
        self.list_actions_taken = []

        if environment_name == 'workflow':
            self.env = workflow_env(number_inputs=self.initial_nodes, node_types=self.node_types,
                                max_number_nodes=self.max_number_nodes)
            #print('w env')
        if environment_name =='single_action':
            self.env = workflow_env(number_inputs=self.initial_nodes, node_types=self.node_types,
                                    max_number_nodes=self.max_number_nodes)
            #print('single env')
        else:
            self.env = environment(number_inputs=self.initial_nodes, node_types=self.node_types,
                                    max_number_nodes=self.max_number_nodes)
            #print('standard env')
        self.env.reset()


        self.success_val = []

    def epsilon_greedy(self):
        pass


    def greedy(self):
        # print('greedystart')
        self.env.reset()
        # self.env = workflow_env([1, 2, 0], node_types=3, max_number_nodes=7)  # reset workflow_env
        for e in range(self.episode_length):
            if self.env.reward > 0.0:
                # print('amazing')
                self.success_val.append(1.0)
                break
            if len(self.env.eligible_actions) == 0:
                # print('no more possible actions')
                self.success_val.append(0.0)
                break
            action = random.choice(self.env.eligible_actions)
            self.env + action
        # print(self.env.state, self.env.reward, self.env.features)

    @staticmethod
    def get_state(state,features):
        in_adj_mat = torch.tensor(state, dtype=torch.float32)
        out_adj_mat = torch.t(in_adj_mat)
        v_feat = torch.tensor(features, dtype=torch.float32)
        return (v_feat, in_adj_mat, out_adj_mat)
