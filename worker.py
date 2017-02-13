import numpy as np
from model import build_model, build_model_checkpoints
from time import time
import scipy.signal
import cPickle

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype('float32')


def multinomial(probs, sample=True):
    if sample:
        probs = probs - np.finfo(np.float32).epsneg
        histogram = np.random.multinomial(1, probs)
        return histogram.argmax()
    else:
        return probs.argmax()


def rollout(env, prob_fn, val_fn, n_steps=20, gamma=0.99):
    n_actions = env.action_space.n
    q_values = np.zeros(n_actions, dtype='float32')

    #checkpoint_id = env.create_checkpoint()
    for a in xrange(n_actions):
        env.create_checkpoint()
        total_reward = 0
        cur_gamma = 1
        #do action for rollout
        s, r, t, r_plus = env.step(a)
        total_reward += cur_gamma*(r + r_plus)
        cur_gamma *= gamma

        #perform rollout if not in terminal state
        if not t:
            for _ in xrange(n_steps):
                #choose and do action
                probs = prob_fn([s])[0]
                action = multinomial(probs)
                s, r, t, r_plus = env.step(action)
                total_reward += cur_gamma * (r + r_plus)
                cur_gamma *= gamma

                if t:
                    break

            # if not terminal do boostrap
            if not t:
                total_reward += cur_gamma * val_fn([s])[0]

        #add reward
        q_values[a] = total_reward
        env.load_from_checkpoint()

    # clear checkpoint and return q values
    return q_values


def set_theano_weights(weights_shared, params):
    for w, p in zip(weights_shared, params):
        p.set_value(w)


def update_shared_weights(weights_shared, steps):
    for w, s in zip(weights_shared, steps):
        w -= s


def run_checkpoint(process, env, weights_shared, global_step, best_reward,
                   weights_save_intervsal, num_test_episodes,
                   n_steps=20, max_steps=50000, gamma=0.99):

    # init model
    steps_fn, prob_fn, val_fn, params = build_model_checkpoints(env.observation_space.shape, env.action_space.n)

    # set initial weights
    set_theano_weights(weights_shared, params)
    current_episode = 0
    start = time()
    while global_step.value < max_steps:
        state = env.reset()

        total_reward = 0
        total_rplus = 0
        step = 0
        current_episode += 1
        while True:

            # state and rewards for training
            q_vals = rollout(env, prob_fn, val_fn, n_steps, gamma)

            # training step
            steps = steps_fn([state], [q_vals])
            update_shared_weights(weights_shared, steps)
            set_theano_weights(weights_shared, params)
            global_step.value += 1

            # do action
            probs = prob_fn([state])[0]
            action = multinomial(probs)
            state, reward, terminal, r_plus = env.step(action)
            total_reward += reward
            step += 1
            total_rplus += r_plus

            if terminal:
                break
        if process == 0 and current_episode % weights_save_intervsal == 0:
            sampled_total_rewards = []
            for ep in range(num_test_episodes):
                total_reward = 0
                state = env.reset()
                while True:
                    probs = prob_fn([state])[0]
                    action = multinomial(probs)
                    state, reward, terminal, r_plus = env.step(action)
                    total_reward += reward
                    if terminal:
                        break
                sampled_total_rewards.append(total_reward)
            reward_mean = sum(sampled_total_rewards) / len(sampled_total_rewards)
            print ('test reward mean', reward_mean)
            if reward_mean > best_reward.value:
                best_reward.value = reward_mean
                param_values = [p.get_value() for p in params]
                with open('weights_steps_{}_reward_{}.pkl'.format(global_step.value, int(reward_mean)), 'wb') as f:
                    cPickle.dump(param_values, f, -1)


        print 'Global step: {}, steps/sec: {:.2f}, episode length {}, r_plus {:.2f}, reward: {:.2f}, best reward: {:.2f}'. \
            format(global_step.value, 1. * global_step.value / (time() - start), step, total_rplus, total_reward, best_reward.value)


def run(process, env, weights_shared, global_step, best_reward,
        weights_save_intervsal, num_test_episodes,
        n_steps=20, max_steps=50000, gamma=0.99):

    # init model
    steps_fn, prob_fn, val_fn, params = build_model(env.observation_space.shape, env.action_space.n)

    # set initial weights
    set_theano_weights(weights_shared, params)

    # state and rewards for training
    states = []
    actions = []
    rewards = []

    epoch = 0
    current_episode = 0
    start = time()
    while global_step.value < max_steps:
        state = env.reset()

        total_reward = 0.
        total_rplus = 0.
        terminal = False
        step = 0
        mean_val = 0.

        while not terminal:

            for _ in xrange(n_steps):
                # do action
                probs = prob_fn([state])[0]
                action = multinomial(probs)

                states.append(state)
                actions.append(action)

                state, reward, terminal, r_plus = env.step(action)
                total_reward += reward
                total_rplus += r_plus
                step += 1

                rewards.append(reward + r_plus)

                if terminal:
                    break

            if terminal:
                rewards.append(0)
            else:
                rewards.append(val_fn([state])[0])

            v_batch = discount(rewards, gamma)[:-1]
            # training step
            steps = steps_fn(states, v_batch, actions)
            update_shared_weights(weights_shared, steps)
            set_theano_weights(weights_shared, params)
            global_step.value += len(rewards) - 1
            mean_val += val_fn([states[len(states)/2]])[0]

            # clear buffers
            del states[:]
            del rewards[:]
            del actions[:]

            if terminal:
                epoch += 1
                break

        current_episode += 1
        if process == 0 and current_episode % weights_save_intervsal == 0:
            sampled_total_rewards = []
            for ep in range(num_test_episodes):
                total_reward = 0
                state = env.reset()
                while True:
                    probs = prob_fn([state])[0]
                    action = multinomial(probs)
                    state, reward, terminal, r_plus = env.step(action)
                    total_reward += reward
                    if terminal:
                        break
                sampled_total_rewards.append(total_reward)
                reward_mean = sum(sampled_total_rewards) / len(sampled_total_rewards)
            print ('test reward mean', reward_mean)
            if reward_mean > best_reward.value:
                best_reward.value = reward_mean
                param_values = [p.get_value() for p in params]
                with open('weights_steps_{}_reward_{}.pkl'.format(global_step.value, int(reward_mean)), 'wb') as f:
                    cPickle.dump(param_values, f, -1)

        report_str = 'Global step: {}, steps/sec: {:.2f}, mean value: {:.2f},  episode length {}, r_plus {:.2f}, reward: {:.2f}, best reward: {:.2f}'.\
            format(global_step.value, 1.*global_step.value/(time() - start), mean_val/step, step, total_rplus, total_reward, best_reward.value)
        print report_str

        if epoch % 1 == 0 and process == 0:
            with open('report.txt', 'a') as f:
                f.write(report_str + '\n')
