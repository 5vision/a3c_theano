import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = 'device=cpu'

import argparse
import worker
from multiprocessing import Process, Array, Value
from model import build_model, build_model_checkpoints
from envs import create_env
from ctypes import c_float
import numpy as np


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--num_workers', default=8, type=int, help="Number of workers.")
parser.add_argument('--env', type=str, default="pong", help="Environment id.")
parser.add_argument('--version', type=str, default="v0", help="Version of environment as in openAI gym.")
parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for reward.")
parser.add_argument('--max_steps', type=int, default=100000000, help="Number of epochs.")
parser.add_argument('--n_steps', type=int, default=100, help="Number of  rollout steps.")
parser.add_argument('--checkpoints', action='store_true', help="run a3c with checkpoins.")
parser.add_argument('--exploration_bonus', action='store_true', help='Add exploration bonus to reward.')
parser.add_argument('--weights_save_interval', default=20, type=int, help="Period testing and saving weighs.")
parser.add_argument('--num_test_episodes', type=int, default=10, help="Number of test episodes.")
#def init_shared_weights(params):
#    weights_shared = []
#    for p in params:
#        weights_shared.append(p.get_value().ravel())
#    w = Array(c_float, np.concatenate(weights_shared))
#    return np.ctypeslib.as_array(w.get_obj())


def init_shared_weights(state_shape, n_actions, checkpoints):
    if checkpoints:
        _, _, _, params = build_model_checkpoints(state_shape, n_actions)
    else:
        _, _, _, params = build_model(state_shape, n_actions)
    weights_shared = []
    for p in params:
        p_shape = p.get_value().shape
        print p_shape
        w = Array(c_float, p.get_value().ravel())
        weights_shared.append(np.ctypeslib.as_array(w.get_obj()).reshape(p_shape))
    return weights_shared


def main():
    args = parser.parse_args()
    weights_save_interval = args.weights_save_interval
    num_test_episodes = args.num_test_episodes

    env = create_env(args.env, 4, args.version, args.exploration_bonus)

    weight_shared = init_shared_weights(env.observation_space.shape,
                                        env.action_space.n, args.checkpoints)
    global_step = Value('i', 0)
    best_reward = Value('f', -1e8)

    # start workers
    if args.checkpoints:
        target_fn = worker.run_checkpoint
    else:
        target_fn = worker.run
    workers = []
    for i in xrange(args.num_workers):
        w = Process(target=target_fn,
                    args=(i, env, weight_shared, global_step, best_reward,
                          weights_save_interval, num_test_episodes,
                          args.n_steps, args.max_steps, args.gamma)
                    )
        w.daemon = True
        w.start()
        workers.append(w)

    # end all processes
    for w in workers:
        w.join()


if __name__ == '__main__':
    main()
