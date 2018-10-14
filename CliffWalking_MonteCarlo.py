import gym
import numpy as np
import time, pickle

def play_episode(env, policy, eps, ACTIONS, render=False):
    state = env.reset()
    s_a_r = []
    ttl_reward = 0
    while True:
        if render: env.render()
        
        action = policy[state] if np.random.random() > eps else np.random.choice(ACTIONS)

        new_state, reward, done, _ = env.step(action)
        ttl_reward += reward

        s_a_r.append((state, action, reward))
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
    GAME = "CliffWalking-v0"
    model = "Monte_Carlo"
    env = gym.make(GAME)
    STATES = [i for i in range(env.nS)]
    ACTIONS = [int(i) for i in range(env.nA)]
    TESTS = 1
    RENDER = True


    #there may be a relationship between epsilon and reward
    #reward based epsilon?
    #this is good but only sometimes
    # it may take long but does it always converge?
    # fixed epsilon?
    EPISODES = 3000
    START_EPSILON = 0.9
    EPSILON_TAPER = 0.003 #0.003 is good when reward consistenly increases
    MIN_EPS = 0 #does this have a place? 0.4 works to get a score of approx -80
    # doesnt curtail many bad rewards :( or does it?
    # this game has very peculiar behavior
    GAMMA = 0.9

    navg = 100
    total_reward = 0
    best = float("-inf")
    
    policy = {state: np.random.choice(ACTIONS) for state in STATES}
    q = {(state, action):0 for state in STATES for action in ACTIONS}
    returns = {(state, action): [] for state in STATES for action in ACTIONS}
    
    start_time = time.time()

    for episode in range(1, EPISODES+1):
        eps = max(START_EPSILON/(1 + EPSILON_TAPER * episode), MIN_EPS)
        #eps = START_EPSILON * (EPISODES-episode)/EPISODES # this one is worse
        if episode % navg == 0:
            avg = total_reward/navg
            print("Episode =", episode, " |  Avg Reward =", avg, " | Epsilon =", eps)
            total_reward = 0

        s_a_r, game_reward = play_episode(env, policy, eps, ACTIONS, render=False)
        s_a_G = find_s_a_G(s_a_r, GAMMA)
        total_reward += game_reward
        if game_reward > best: best = game_reward
        
        seen = []
        for (s, a, G) in s_a_G:
            if (s, a) not in seen:
                seen.append((s, a))
                returns[(s, a)].append(G)
                q[(s, a)] = np.mean(returns[s, a])

        for s in STATES: policy[s] = best_action_and_value(q, s, ACTIONS)[0]
    
    end_time = time.time()
    print("Total training time:", end_time - start_time, "Best Score:", best)
    
    #for i in range(TESTS): play_episode(env, policy, 0, ACTIONS, render=True)
    env.close()
    path = "saves/" + GAME + "_" + model + "_policy.pkl" 
    with open(path, "wb+") as fp: pickle.dump(policy, fp)