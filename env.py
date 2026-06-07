import random
from pettingzoo.classic import texas_holdem_no_limit_v6

# Initialize the environment
env = texas_holdem_no_limit_v6.env(render_mode="human")
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        # observation['observation'] contains the vectorized state (cards + betting history)
        # Use action_mask to only pick from legal actions
        action_mask = observation['action_mask']
        legal_actions = [i for i, valid in enumerate(action_mask) if valid]
        action = random.choice(legal_actions)  # Replace with your DQN policy
        
    env.step(action)