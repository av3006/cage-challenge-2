from collections import deque


class RewShapeWrapper():
    def __init__(self, env=None, wt=1):
        self.action_buffer = deque(maxlen=2)
        self.observation_buffer = deque(maxlen=2)
        self.env = env
        self.TRUE_GOAL = 24
        self.last_huer = self.TRUE_GOAL # goal distance -> 1+2+3+3*(1+2+3)
        self.weight = wt

    def get_table(self):
        # get table from parent wrapper
        return self.env.get_table()
    
    def calc_huer(self):
        # get a hueristic for the current state of the environment
        # from the table
        tb = self.get_table()
        huer_score = 0
        max_state = [0]*3

        def calc_single(base_multi, known, access):
            score = 0
            if known == 'True':
                score += (1*base_multi)
            if access == 'User':
                score += (2*base_multi)
            elif access == 'Privileged':
                score += (3*base_multi)
            return score

        for row in tb:
            row.border = False
            row.header = False
            hostname = row.get_string(fields=["Hostname"]).strip()
            known = row.get_string(fields=["Known"]).strip()
            access = row.get_string(fields=["Access"]).strip()

            if hostname[:-1] == 'User':
                curr_score = calc_single(1, known, access)
                max_state[0] = max(max_state[0], curr_score)
            elif hostname[:-1] == 'Enterprise':
                curr_score = calc_single(2, known, access)
                max_state[1] = max(max_state[1], curr_score)
            elif hostname[:-1] == 'Op_Host':
                curr_score = calc_single(3, known, access)
                max_state[2] = max(max_state[2], curr_score)
        
        huer_score = self.TRUE_GOAL - sum(max_state) # distance to min goal
        
        return huer_score
    
    
    def step(self, action=0):
        obs, rew, done, info = self.env.step(action=action)

        self.action_buffer.append(action)
        self.observation_buffer.append(obs)
        # basic reward shaping added to punish agent for repeating actions
        # and to reward agent for taking actions that change the environment
        if len(self.action_buffer) == 2:
            if list(self.observation_buffer[0]) == list(self.observation_buffer[1]): # if repeat obs 
                if self.action_buffer[0] == self.action_buffer[1]: # if also repeat action
                    rew -= 0.1 # punish heavily
                else:
                    rew -= 0.05 # punish lightly
        # reward shaping reward traj 
        new_huer = self.calc_huer()
        delta = new_huer - self.last_huer # delta will be negative (new distance is less) if improved
        delta = -1*delta # make positive
        rew += (delta*self.weight)
        self.last_huer = new_huer
        
        return obs, rew, done , info
    
    def reset(self):
        obs = self.env.reset()
        self.action_buffer = deque(maxlen=2)
        self.observation_buffer = deque(maxlen=2)
        self.observation_buffer.append(obs)
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