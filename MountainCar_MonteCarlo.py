import gym
import numpy as np
import time, pickle

#WIP not able to complete MountainCar yet. Probably hyperparameter tuning since
#model has some success in CliffWalking
def MountainCar_states(): # returns states, postions, velocitys
    # make sure to change round(_, DECIMAL_PLACES) when changing states
    position_states, velocity_states = 18, 14 #18, 14
    position, velocity = [-1.2, 0.6], [-0.07, 0.07] # [min, max]
    position_range, velocity_range = position[1] - position[0], velocity[1] - velocity[0] #1.8, 0.14
    
    positions = [round(position[0] + i * position_range/position_states, 2) for i in range(position_states+1)]
    velocitys = [round(velocity[0] + i * velocity_range/velocity_states, 3) for i in range(velocity_states+1)]
    
    states = [(p, v) for p in positions for v in velocitys]
    return states, positions, velocitys

def round_state(state, tp, tv): #returns rounded state = [p, v]
    def round_p(p, tp=tp):
        d = [abs(i-p) for i in tp]
        return tp[np.argmin(d)]
    def round_v(v, tv=tv):
        d = [abs(i-v) for i in tv]
        return tv[np.argmin(d)]
    
    return (round_p(state[0]), round_v(state[1]))

def play_episode(env, policy, eps, ACTIONS, tp, tv, render=False):
    state = env.reset()
    s_a_r = []
    ttl_reward = 0
    while True:
        if render: env.render()
        
        action = policy[round_state(state, tp, tv)] if np.random.random() > eps else np.random.choice(ACTIONS)

        new_state, reward, done, _ = env.step(action)
        ttl_reward += reward

        s_a_r.append((round_state(state, tp, tv), action, reward))
        state = new_state
        if done: break

    return s_a_r, ttl_reward

def find_s_a_G(s_a_r, GAMMA):
    G = 0
    s_a_G = []
    for (s, a, r) in reversed(s_a_r):
        s_a_G.append((s, a,  G))
        G = r + GAMMA * G
    return reversed(s_a_G)
    
def best_action_and_value(q, s, ACTIONS):
    best_value = float("-inf")
    for a in ACTIONS:
        if q[s, a] > best_value:
            best_value = q[s, a]
            best_action = a
    return best_action, best_value

if __name__ == "__main__":
    GAME = "MountainCar-v0"
    model = "Monte_Carlo"
    env = gym.make(GAME)
    STATES, tp, tv = MountainCar_states()
    ACTIONS = (0, 1, 2) #(LEFT, COAST, RIGHT)
    TESTS = 2
    RENDER = True

    EPISODES = 2000
    START_EPSILON = 0.9 # lots of hyperparameter tuning required
    EPSILON_TAPER = 0.001
    MIN_EPS = 0
    GAMMA = 0.9

    navg = 100
    total_reward = 0
    best = -201
    
    policy = {state: np.random.choice(ACTIONS) for state in STATES}
    q = {(state, action):0 for state in STATES for action in ACTIONS}
    returns = {(state, action): [] for state in STATES for action in ACTIONS}
    
    start_time = time.time()

    for episode in range(EPISODES+1):
        eps = max(START_EPSILON/(1 + EPSILON_TAPER * episode), MIN_EPS)

        if episode % navg == 0:
            avg = total_reward/navg
            print("Episode =", episode, " |  Avg Reward =", avg, " | Epsilon =", eps)
            total_reward = 0

        #if episode % 1000 == 0: play_episode(env, policy, 0, ACTIONS, tp, tv, render=True)

        s_a_r, game_reward = play_episode(env, policy, eps, ACTIONS, tp, tv, render=False)
        s_a_G = find_s_a_G(s_a_r, GAMMA)
        total_reward += game_reward
        if game_reward > best: best = game_reward
        
        seen = []
        for (s, a, G) in s_a_G:
            if (s, a) not in seen:
                seen.append((s, a))
                returns[(s, a)].append(G)
                q[(s, a)] = np.mean(returns[(s, a)])

        for s in STATES: policy[s] = best_action_and_value(q, s, ACTIONS)[0]
    
    end_time = time.time()
    print("Total training time:", end_time - start_time, "Best Reward:", best)
    
    for i in range(TESTS): play_episode(env, policy, 0, ACTIONS, tp, tv, render=True)
    env.close()
    path = "saves/" + GAME + "_" + model + "_policy.pkl" 
    #with open(path, "wb+") as fp: pickle.dump(policy, fp)
