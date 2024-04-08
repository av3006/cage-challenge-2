from collections import deque
import numpy as np

class RewShapeBlue():
    def __init__(self, env=None, wt=1, max_steps = 30):
        self.env = env
        self.last_huer = self.calc_huer()
        self.weight = wt
        self.step_count = 0
        self.max_steps = max_steps 

    def get_table(self):
        # get table from parent wrapper
        return self.env.get_table()
    
    def calc_huer(self):
        # get a hueristic for the current state of the environment
        # from the table
        tb = self.get_table()
        huer_score = 0

        for row in tb._rows:
            # Host
            hostname = row[2]
            base = 0
            if hostname[:-1] == 'User':
                base = 1
            elif hostname[:-1] == 'Enterprise':
                base = 2
            elif hostname[:-1] == 'Op_Host':
                base = 3
            # Activity
            activity = row[3]
            a_value = 0
            if activity == 'None':
                a_value = 0
            elif activity == 'Scan':
                a_value = 1
            elif activity == 'Exploit':
                a_value = 2
            else:
                raise ValueError('Table had invalid Access Level')

            # Compromised
            compromised = row[4]
            c_value = 0
            if compromised == 'No':
                c_value = 0
            elif compromised == 'Unknown':
                c_value = 1
            elif compromised == 'User':
                c_value = 2
            elif compromised == 'Privileged':
                c_value = 3
            else:
                raise ValueError('Table had invalid Access Level')

            huer_score += base*(a_value + c_value)

        t_scale = 2*(1- (self.step_count/float(self.max_steps)))
        t_weight = np.tanh(t_scale)
        return huer_score * t_weight
    
    def step(self, action=0):
        obs, rew, done, info = self.env.step(action=action)
        # reward shaping reward traj 
        new_huer = self.calc_huer()
        delta = new_huer - self.last_huer # delta will be negative (new distance is less) if improved
        delta = -1*delta # make positive
        rew += (delta*self.weight)
        self.last_huer = new_huer
        self.step_count += 1

        return obs, rew, done , info
    
    def reset(self):
        obs = self.env.reset()
        self.last_huer = 0
        self.step_count = 0
        self.last_huer = self.calc_huer()
        return obs

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self,agent:str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self,agent):
        return self.env.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.env.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.env.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.env.get_attr('get_reward_breakdown')(agent)