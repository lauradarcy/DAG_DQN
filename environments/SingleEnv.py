import numpy as np
import networkx as nx
import random
import networkx.algorithms.isomorphism as iso
import operator


class single_env:
    def __init__(self,number_inputs=[0], node_types=1, max_number_nodes=4,loss=0):
        self.number_inputs = len(number_inputs)
        self.node_types = node_types
        self.state = np.zeros((self.number_inputs,self.number_inputs))
        self.features = []
        self.max_number_nodes = max_number_nodes

        self.loss = loss
        self.workflow_vals = [1]
        self.workflow_vals_truth = [1]

        self.action_space = self.create_action_space(all_vals=True)
        self.eligible_actions = self.create_action_space()

        self.reward = 0
        self.terminated = False

        self.truth_state = np.zeros((self.number_inputs,self.number_inputs))
        self.truth_features = []
        self.prev_action = None

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

    def create_action_space(self,all_vals=False):
        action_space = []
        if np.shape(self.state)[0]<self.max_number_nodes or all_vals==True:
            if (np.shape(self.state)[0]>1 and np.sum(self.state, axis=0)[-1]>0) or all_vals==True or np.shape(self.state)[0]==1:
                #print('self.state',self.state,np.sum(self.state, axis=0)[-1],(np.shape(self.state)[0]>1 and np.sum(self.state, axis=0)[-1]>0),all_vals==True,np.shape(self.state)[0]==1)
                for node_type in range(self.node_types):
                    action_space.append(('N',node_type))
        if np.shape(self.state)[0]>self.number_inputs:
            for i in range(np.shape(self.state)[0]-1):
                if self.state[i,-1]>0 and all_vals==False:
                    continue
                else:
                    action_space.append(('E',i))
        #action_space.append(('T',0))
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
        graph_current = single_env.create_DAG(state,features)
        graph_truth = single_env.create_DAG(self.truth_state,self.truth_features)
        nm = iso.numerical_node_match('feat', range(self.node_types))

        if nx.is_isomorphic(graph_current,graph_truth,node_match=nm):
            reward=1000
        elif self.loss != 0:
            reward=-1
        '''if workflow_vals[-1] == self.workflow_vals_truth[-1]:
            #print('WOW')
            reward = 1000'''
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
        action_type = action[0]
        if action_type == 'N':
            if np.shape(self.state)[0]==self.max_number_nodes:
                raise ValueError('max number of nodes ({}) already reached, remaining actions are adding edges and terminating episode'.format(self.max_number_nodes))
            node_type = action[1]
            if node_type >= self.node_types:
                raise ValueError('{} is not a valid node type, please select a value of 0 through {} inclusive'.format(node_type,self.node_types-1))
            connections = np.zeros((np.shape(self.state)[0], 1))
            lower_axis = np.zeros((1, np.shape(self.state)[0] + 1))
            self.state = np.concatenate((self.state, connections), axis=1)
            self.state = np.concatenate((self.state, lower_axis), axis=0)
            self.features.append(self.encode_features(node_type))
            self.prev_action = action
        elif action_type == 'E':
            from_node = action[1]
            valid_choices = np.shape(self.state)[0]-2
            if from_node > valid_choices:
                raise ValueError('{} not a valid edge, current valid choices are 0 through {} inclusive'.format(from_node, valid_choices))
            self.state[from_node,-1]=1
            self.prev_action = action
            '''elif action_type == 'T':
            self.terminated = True
            self.reward = self.determine_reward()
            self.prev_action = action'''
        else:
            raise ValueError('{} not a valid action type. see self.action_space for list of possible actions'.format(action_type))
        self.reward = self.determine_reward(self.state, self.features, self.workflow_vals)
        self.eligible_actions = self.create_action_space()

    def __sub__(self, action):
        if action is None:
            print('no prev action')
        elif action[0] == 'N':
            self.state = np.delete(self.state, -1, 1)
            self.state = np.delete(self.state, -1, 0)
            self.features.pop(-1)
        elif action[0] == 'E':
            self.state[action[1],-1]=0
        elif action[0] == 'T':
            self.terminated = False
            self.reward = 0
        self.prev_action = None
        self.reward = self.determine_reward(self.state, self.features, self.workflow_vals)
        self.eligible_actions = self.create_action_space()


    def take_action(self, state, features, action):
        new_features = features.copy()
        new_state = state.copy()

        action_type = action[0]
        if action_type == 'N':
            if np.shape(new_state)[0]==self.max_number_nodes:
                raise ValueError('max number of nodes ({}) already reached, remaining actions are adding edges and terminating episode'.format(self.max_number_nodes))
            node_type = action[1]
            if node_type >= self.node_types:
                raise ValueError('{} is not a valid node type, please select a value of 0 through {} inclusive'.format(node_type,self.node_types-1))
            connections = np.zeros((np.shape(new_state)[0], 1))
            lower_axis = np.zeros((1, np.shape(new_state)[0] + 1))
            new_state = np.concatenate((new_state, connections), axis=1)
            new_state = np.concatenate((new_state, lower_axis), axis=0)
            new_features.append(self.encode_features(node_type))

        elif action_type == 'E':
            from_node = action[1]
            valid_choices = np.shape(new_state)[0]-2
            if from_node > valid_choices:
                raise ValueError('{} not a valid edge, current valid choices are 0 through {} inclusive'.format(from_node, valid_choices))
            new_state[from_node,-1]=1

        return new_state, new_features


'''a = workflow_env()
print(a.truth_state)
print(a.workflow_vals_truth)
a+(0,1)
print(operator.ifloordiv(7,2))'''