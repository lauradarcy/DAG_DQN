from agents.network.DQN import DQN

from environments.Environment import environment
from environments.WorkflowEnvironment import workflow_env
from environments.SingleEnv import single_env
import torch
import random


import torch.optim as optim

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class S_Agent:
    def __init__(self, input_node_type=[0], graph_size=4, node_types=1, environment_name=None, total_episodes=None):
        #print('standard')
        self.max_number_nodes = graph_size
        self.node_types = node_types

        self.initial_nodes = input_node_type

        self.epsilon = 0.5
        self.epsilon_delta = (.05/total_episodes)
        self.gamma = 0.9
        self.episode_length = 30


        if environment_name == 'workflow':
            self.env = workflow_env(number_inputs=self.initial_nodes, node_types=self.node_types,
                                max_number_nodes=self.max_number_nodes)
            #print('w env')
        if environment_name =='single':
            self.env = single_env(number_inputs=self.initial_nodes, node_types=self.node_types,
                                    max_number_nodes=self.max_number_nodes)
            #print('single env')
        else:
            self.env = environment(number_inputs=self.initial_nodes, node_types=self.node_types,
                                    max_number_nodes=self.max_number_nodes)
            #print('standard env')

        D_in = self.env.node_types
        H1 = 20
        H2 = 20
        H3 = 20
        H4 = 20
        H5 = 1
        learning_rate = 1e-3
        weight_decay_val = 0.0

        self.NN = DQN(D_in, H1, H2, H3, H4, H5)

        if torch.cuda.is_available():
            self.cuda = True
            self.NN = self.NN.cuda()
        else:
            self.cuda = False

        self.optimizer = optim.Adam(self.NN.parameters(), lr=learning_rate, weight_decay=weight_decay_val)
        self.success_val = []

    def epsilon_greedy(self):
        self.env.reset()

        self.epsilon -= self.epsilon_delta

        for e in range(self.episode_length):
            action = self.epsilon_greedy_action_selection()
            self.env + action

            q_val = self.compute_Q()

            step_q = self.compute_bellman_Q()

            loss = (q_val-step_q).pow(2) / 2.0
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
            if self.env.reward > 0.0:
                break

            if len(self.env.eligible_actions) == 0:
                break
        else:
            print('end episode, somehow took {} steps??'.format(self.episode_length))

    def epsilon_greedy_action_selection(self):
        if random.random() <= self.epsilon:
            return random.choice(self.env.eligible_actions)
        else:
            return self.choose_maximum_action()[0]

    def choose_maximum_action(self):
        self.NN.eval()
        max_q = -float('Inf')
        actions_list = self.env.eligible_actions.copy()
        random.shuffle(actions_list)
        best_action = actions_list[0]
        amount_to_sample = len(actions_list)
        for i in range(amount_to_sample):
            # search whole action space or to max sample search whichever is smaller
            action = actions_list[i]
            self.env + action
            q = self.compute_Q()
            if q > max_q:
                max_q = q
                best_action = action
            self.env - action
        self.NN.train()
        return best_action, max_q

    def compute_bellman_Q(self):
        r = self.env.reward
        if len(self.env.eligible_actions) > 0:
            _, q_max = self.choose_maximum_action()
        else:
            q_max = 0
        return r + (self.gamma * q_max)

    def compute_Q(self):
        state = S_Agent.get_state(self.env)
        if self.cuda:
            state = (s.cuda() for s in state)
        q = self.NN(state)
        return q

    def greedy(self):

        self.env.reset()

        for e in range(self.episode_length):
            if self.env.reward > 0.0:
                self.success_val.append(1.0)
                break
            if len(self.env.eligible_actions) == 0:
                # print('no more possible actions')
                self.success_val.append(0.0)
                break
            a_max, _ = self.choose_maximum_action()
            self.env + a_max
        # print(self.env.state, self.env.reward, self.env.features)

    @staticmethod
    def get_state(environment):
        in_adj_mat = torch.tensor(environment.state, dtype=torch.float32)
        out_adj_mat = torch.t(in_adj_mat)
        v_feat = torch.tensor(environment.features, dtype=torch.float32)
        return (v_feat, in_adj_mat, out_adj_mat)

