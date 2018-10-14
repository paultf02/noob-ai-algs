import gym
import numpy as np
import time, pickle

#used for discretizing states
def MountainCar_states(): # returns states, postions, velocitys
    # make sure to change round(_, DECIMAL_PLACES) when changing number of states
    position_states, velocity_states = 18, 14 
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

def play_episode(env, policy): #plays an epsiode using policy
    s = env.reset()
    while True:
        env.render()
        s, reward, done, _ = env.step(policy[round_state(s, tp, tv)])
        if done: break

def best_action_and_value(q, s): #returns best_action, best_value 
    best_v = float("-inf")
    for a in ACTIONS:
        if q[s, a] > best_v:
            best_v = q[s, a]
            best_a = a
    return best_a, best_v

if __name__ == '__main__':
    GAME = "MountainCar-v0"
    model = "Q_Learning"
    #ACTIONS = (0, 1, 2) #(LEFT, COAST, RIGHT)
    STATES, tp, tv = MountainCar_states()
    env = gym.make(GAME)

    #hyperparams need to be tuned
    TESTS = 3
    EPISODES = 5000
    START_EPSILON = 1
    EPSILON_TAPER = 0.01
    START_ALPHA = 0.8
    ALPHA_TAPER = 0.01 
    GAMMA = 0.95
    navg = 100
    
    policy = {state: np.random.choice(ACTIONS) for state in STATES}
    q = {(state, action): 0 for state in STATES for action in ACTIONS}
    state_count = {state: 0 for state in STATES}
    total_reward = 0
    best = -201

    start_time = time.time()
    for episode in range(EPISODES):
        eps = START_EPSILON/(1 + EPSILON_TAPER * episode)

        if episode % navg == 0:
            avg = total_reward/navg
            print("Episode =", episode, " |  Avg Reward =", avg, " | Epsilon =", eps)
            total_reward = 0


        this_reward = 0
        cur_state = round_state(env.reset())
        while True:
            prev_state = cur_state
            state_count[prev_state] = state_count.get(prev_state, 0) + 1

            #epsilon greedy action
            action = best_action_and_value(q, prev_state)[0] if np.random.random() > eps else np.random.choice(ACTIONS)

            cur_state, reward, done, _ = env.step(action)
            cur_state = round_state(cur_state, tp, tv)
            total_reward += reward
            this_reward += reward
            
            # updating q table
            alpha = START_ALPHA/(1+ALPHA_TAPER*state_count[prev_state])
            q_sprime = best_action_and_value(q, cur_state)[1] # max q(s_prime, a) Q-LEARNING
            #q_sprime = q[cur_state, policy[cur_state]] # q(s_prime, policy[state]) SARSA 
            q[prev_state, action] = q[prev_state, action] + alpha * (reward + GAMMA*q_sprime - q[prev_state, action])
            
            if done: break
        
        if this_reward > best:
                policy = {state: best_action_and_value(q, state)[0] for state in STATES}
                best, best_q, best_p = this_reward, q, policy

    end_time = time.time()
    print("Training time:", end_time - start_time, "Best reward = ", best)

    final_policy = {state: best_action_and_value(q, state)[0] for state in STATES}
    for i in range(TESTS): play_episode(env, final_policy)
    env.close()

    path = "saves/" + GAME + "_" + model + "_policy.pkl" 
    with open(path, "wb+" ) as fp: pickle.dump(best_p, fp)