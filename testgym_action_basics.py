import gym
env = gym.make("CartPole-v0")

#print(env.action_space)

#print(env.observation_space)

observation = env.reset()

for t in range(1000):
    env.render()

    cart_pos, cart_val, pole_ang, ang_vel = observation

    if pole_ang>0:
        action = 1
    else:
        action = 0
    
    observation, reward, done, info = env.step(action)
    
