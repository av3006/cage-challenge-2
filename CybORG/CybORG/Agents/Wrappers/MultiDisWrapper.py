from collections import defaultdict
import copy
import inspect, pprint
from typing import Union

from CybORG.Agents.SimpleAgents import BaseAgent
from CybORG.Agents.Wrappers import BaseWrapper
from CybORG.Shared import Results


class MultiDisWrapper(BaseWrapper):
    def __init__(self, env: Union[type, BaseWrapper] = None, agent: BaseAgent = None):
        super().__init__(env, agent)
        self.possible_actions = None
        self.actions_list = []
        self.action_signature = {}
        self.actions_dict = {} 
        self.get_action_space('Red')

    def step(self, agent=None, action= None) -> Results:
        # get the action object from the action index
        act_idx = action[0]
        action_obj = self.actions_list[act_idx]
        params_d = {}
        for i in range(0, len(self.actions_dict[action_obj])):
            p_key = self.actions_dict[action_obj][i][0]
            p_val = self.actions_dict[action_obj][i][1][action[i+1]]
            params_d[p_key] = p_val
        action = action_obj(**params_d)

        return super().step(agent, action)

    def get_action_space(self, agent: str):
        # get the longest params
        max_len = 0
        for key, value in self.actions_dict.items():
            max_len = max(max_len, len(value)+1)
        action_lens = [0]*max_len
        action_lens[0] = len(self.actions_dict)
        for key, value in self.actions_dict.items():
            for i, v in enumerate(value):
                action_lens[i+1] = max(action_lens[i+1], len(v[1]))  
        
        return action_lens

    def action_space_change(self, action_space: dict) -> int:
        assert type(action_space) is dict, \
            f"Wrapper required a dictionary action space. " \
            f"Please check that the wrappers below the ReduceActionSpaceWrapper return the action space as a dict "
        
        possible_actions = []
        temp = {}
        params = ['action']
        actions_dict = {}
        actions_list = []
        # for action in action_space['action']:
        for i, action in enumerate(action_space['action']):
            if action not in self.action_signature:
                self.action_signature[action] = inspect.signature(action).parameters
            param_dict = {}
            param_list = [{}]
            for p in self.action_signature[action]:
                if p == 'priority':
                    continue
                temp[p] = []
                if p not in params:
                    params.append(p)

                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            p_dict[p] = key
                            new_param_list.append({key: value for key, value in p_dict.items()})
                    param_list = new_param_list
            for p_dict in param_list:
                possible_actions.append(action(**p_dict))
            
            param_count = defaultdict(set)
            for p_dict in param_list:
                for key, value in p_dict.items():
                    param_count[key].add(value)
            actions_dict[action] = [(key, list(value)) for key, value in param_count.items()]
            actions_list.append(action)

        self.possible_actions = possible_actions
        self.actions_dict = actions_dict
        self.actions_list = actions_list

        return self.actions_dict
