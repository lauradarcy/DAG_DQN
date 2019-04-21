import numpy as np

import random

import operator


class workflow_env:
    def __init__(self,number_inputs=[0], node_types=1, max_number_nodes=4):

        self.number_inputs = len(number_inputs)
        self.node_types = node_types
        self.state = np.zeros((self.number_inputs,self.number_inputs))
        self.features = []
        self.max_number_nodes = max_number_nodes
        self.workflow_vals = [1]
        self.workflow_vals_truth = [1]

        self.eligible_actions = self.create_action_space()

        self.reward = 0
        self.terminated = False
        self.truth_state = np.zeros((self.number_inputs,self.number_inputs))
        self.truth_features = []

        for node in number_inputs:
            self.features.append(self.encode_features(node))

        self.generate_random_truth_values()

    def generate_random_truth_values(self):
        while len(self.eligible_actions) > 0:
            self+random.choice(self.eligible_actions)
        self.truth_state = self.state.copy()
        self.truth_features = self.features.copy()
        self.workflow_vals_truth = self.workflow_vals.copy()
        self.reset()

    def reset(self):
        self.state = np.zeros((self.number_inputs, self.number_inputs))
        self.features = []
        self.workflow_vals = [1]
        for node in range(self.number_inputs):
            self.features.append(self.encode_features(node))

        self.eligible_actions = self.create_action_space()

        self.reward = 0
        self.terminated = False

    def create_action_space(self, state=None):
        action_space = []
        if state is None:
            state = self.state

        if np.shape(state)[0] < self.max_number_nodes:
            for node_type in range(self.node_types):
                for i in range(1,2 ** (state.shape[0])):
                    action_space.append((node_type,i))
        return action_space

    def encode_features(self,node_type):
        one_hot_encoding = [0 for _ in range(self.node_types)]
        one_hot_encoding[node_type] = 1
        return one_hot_encoding

    def determine_reward(self,state,features,workflow_vals):
        reward = 0

        if np.shape(state)[0] != np.shape(self.truth_state)[0]:
            self.terminated = False
            return reward
        #print('workflowvals and truth',workflow_vals,self.workflow_vals_truth)

        if workflow_vals[-1] == self.workflow_vals_truth[-1]:
            #print('WOW')
            reward = 1000
        self.terminated = True
        return reward

    @staticmethod
    def create_DAG(out_adj, features):
        g = nx.DiGraph()
        for index,node in enumerate(features):
            g.add_node(index, feat=node.index(1))
        for index, value in np.ndenumerate(out_adj):
            if value==1.0:
                g.add_edge(index[0],index[1])
        return g

    def __add__(self, action):
        node_type = action[0]
        list_of_operators = [operator.add,operator.sub,operator.mul,operator.ifloordiv,operator.ipow]
        self.state = np.pad(self.state,(0,1),'constant')
        self.features.append(self.encode_features(node_type))
        edges_to_add = format(action[1],'b').rjust(self.state.shape[0]-1,'0')
        new_node_workflow_total = 0
        for index,edge in enumerate(edges_to_add):
            if edge == '1':
                self.state[index][-1]=1
                new_node_workflow_total += list_of_operators[node_type](self.workflow_vals[index],2)
        self.workflow_vals.append(new_node_workflow_total)
        self.reward = self.determine_reward(self.state,self.features,self.workflow_vals)
        self.eligible_actions = self.create_action_space()

    def __sub__(self, other):
        self.state = np.delete(self.state,-1,1)
        self.state = np.delete(self.state, -1,0)
        self.features.pop(-1)

        self.workflow_vals.pop(-1)

        self.reward = self.determine_reward(self.state,self.features,self.workflow_vals)
        self.eligible_actions = self.create_action_space()

    def take_action(self, state, features, action):
        new_features = features.copy()
        new_state = state.copy()
        node_type = action[0]
        new_state = np.pad(new_state, (0, 1), 'constant')
        new_features.append(self.encode_features(node_type))
        edges_to_add = format(action[1], 'b').rjust(new_state.shape[0] - 1, '0')
        for index, edge in enumerate(edges_to_add):
            if edge == '1':
                new_state[index][-1] = 1
        return new_state, new_features


'''a = workflow_env()
print(a.truth_state)
print(a.workflow_vals_truth)
a+(0,1)
print(operator.ifloordiv(7,2))'''