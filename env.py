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
        # This is where your DQN predicts the action based on the one-hot encoded state
        action = env.action_space(agent).sample() # Replace with your DQN policy
        
    env.step(action)