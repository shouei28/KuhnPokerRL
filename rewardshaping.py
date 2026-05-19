from pettingzoo.utils.wrappers import BaseWrapper

class RewardShapingWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        # Execute the action in the base environment
        super().step(action)
        
        # Intercept the current state and reward
        observation, reward, termination, truncation, info = self.env.last()
        
        # Implement your shaping logic
        # For example, providing a small positive shaping reward for aggressive play
        shaped_reward = reward
        if action == 1: # Assuming 1 is 'Bet'
            shaped_reward += 0.1 
            
        # Update the reward dictionary manually
        self.env.rewards[self.env.agent_selection] = shaped_reward