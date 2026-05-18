from pettingzoo.classic import kuhn_poker_v2

# Initialize the environment
env = kuhn_poker_v2.env(render_mode="human")
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