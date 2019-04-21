from agents.network.DQN import DQN

from environments.Environment import environment
import torch
import random

from agents.Replay_v2 import ExperienceReplay
import torch.optim as optim


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''make it so buffer stores time of memory, sort by unusual and then time, also store best action/best actions state'''

class RL3:
    def __init__(self, input_node_type=[0], graph_size=4, node_types=1, sample_size=0, loss=0):
        self.max_sample_size = sample_size
        self.max_number_nodes = graph_size
        self.node_types = node_types
        self.loss = loss
        self.initial_nodes = input_node_type
        self.training_index = 0
        self.loss_mini_batch = 0
        self.memory = ExperienceReplay(unusual_sample_factor=.4)

        self.epsilon = 0.4
        self.gamma = 0.5
        self.episode_length = 30
        self.list_actions_taken = []

        self.env = environment(number_inputs=self.initial_nodes, node_types=self.node_types,
                               max_number_nodes=self.max_number_nodes, loss=self.loss)
        self.env.reset()

        D_in = self.env.node_types
        H1 = 20
        H2 = 20
        H3 = 20
        H4 = 20
        H5 = 1
        learning_rate = 1e-3
        weight_decay_val = 0.0

        self.NN = DQN(D_in, H1, H2, H3, H4, H5)
        self.TargetNetwork = DQN(D_in, H1, H2, H3, H4, H5)

        if torch.cuda.is_available():
            self.cuda = True
            self.NN = self.NN.cuda()
        else:
            self.cuda = False

        self.optimizer = optim.Adam(self.NN.parameters(), lr=learning_rate, weight_decay=weight_decay_val)
        self.optimizer.zero_grad()
        self.success_val = []

    def epsilon_greedy(self):
        self.env.reset()
        self.list_actions_taken = []
        self.epsilon *= .9995
        self.training_index += 1

        for e in range(self.episode_length):
            action = self.epsilon_greedy_action_selection()
            self.env + action
            '''if len(self.memory.buffer) >=30:
                for (state,features,reward,done,time_step,bellman_action) in self.memory.sample(30):
                    state_tuple = A_Agent.get_state(state, features)
                    q_val = self.compute_Q(state_tuple)
                    step_q = self.bellman_q_replay(reward,state,features,done,bellman_action)
                    loss = (q_val - step_q).pow(2) / 2.0
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()'''


            state_tuple = RL3.get_state(self.env.state, self.env.features)
            q_val = self.compute_Q(state_tuple)

            step_q, bellman_action = self.compute_bellman_Q()
            self.memory + (self.env.state.copy(), self.env.features.copy(), self.env.reward, self.env.terminated,
                           self.training_index, bellman_action)

            loss = (q_val - step_q).pow(2) / 2.0
            # print(type(loss))
            # loss.backward()
            self.loss_mini_batch += loss

            # print(self.training_index)
            # print(type(loss_mini_batch))
            if self.training_index % 1 == 0:
                # print('h')

                self.loss_mini_batch.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.loss_mini_batch = 0
            if self.training_index % 5 == 0:
                self.TargetNetwork.load_state_dict(self.NN.state_dict())
            if self.env.reward != 0.0:
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

    def choose_maximum_action(self, greedy=False):
        self.TargetNetwork.eval()
        max_q = -float('Inf')
        actions_list = self.env.eligible_actions.copy()
        random.shuffle(actions_list)
        best_action = actions_list[0]
        amount_to_sample = len(actions_list) if self.max_sample_size == 0 or greedy is True else min(
            self.max_sample_size, len(actions_list))
        for i in range(amount_to_sample):
            # search whole action space or to max sample search whichever is smaller
            action = actions_list[i]
            self.env + action
            state_tuple = RL3.get_state(self.env.state, self.env.features)
            q = self.compute_Q(state_tuple,self.TargetNetwork)  # what is the value of this state
            if q > max_q:
                max_q = q
                best_action = action
            self.env - action
        self.TargetNetwork.train()
        return best_action, max_q

    def best_action_replay(self,state,features,action):
        self.TargetNetwork.eval()

        new_state, new_features = self.env.take_action(state, features, action)
        state_tuple = RL3.get_state(new_state, new_features)
        q = self.compute_Q(state_tuple, self.TargetNetwork)  # what is the value of this state
        self.TargetNetwork.train()
        return q

    def bellman_q_replay(self,reward,state,features,done,action):
        r = reward
        if done:
            q_max = 0
        else:
            q_max = self.best_action_replay(state,features,action)
        return r + (self.gamma * q_max)

    def compute_bellman_Q(self):
        r = self.env.reward
        if len(self.env.eligible_actions) > 0:
            bellman_action, q_max = self.choose_maximum_action()
        else:
            q_max = 0
            bellman_action = 0
        return r + (self.gamma * q_max), bellman_action

    def compute_Q(self, state_tuple, network=None):
        if network is None:
            network = self.NN
        if self.cuda:
            state_tuple = (s.cuda for s in state_tuple)
        q = network(state_tuple)
        return q

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
            a_max, _ = self.choose_maximum_action(greedy=True)
            self.env + a_max
        # print(self.env.state, self.env.reward, self.env.features)

    @staticmethod
    def get_state(state,features):
        in_adj_mat = torch.tensor(state, dtype=torch.float32)
        out_adj_mat = torch.t(in_adj_mat)
        v_feat = torch.tensor(features, dtype=torch.float32)
        return (v_feat, in_adj_mat, out_adj_mat)
