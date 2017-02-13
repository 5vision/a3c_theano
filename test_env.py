from envs import create_env
import argparse
from model import build_model
from worker import multinomial
import cPickle

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--episodes', default=10, type=int, help="Number of episodes to test.")
parser.add_argument('--env', type=str, default="skiing", help="Environment id.")
parser.add_argument('--version', type=str, default="v0", help="Version of environment as in openAI gym.")
parser.add_argument('--n_frames', type=int, default=4, help="Number of frames to stack.")
parser.add_argument('--model', type=str, default=4, help="Path to file with model.")
parser.add_argument('--sample', action='store_true', help="Sample action insted of argmax of probs.")
args = parser.parse_args()


def test():
    env = create_env(args.env, args.n_frames, args.version)

    _, prob_fn, _, params = build_model(env.observation_space.shape, env.action_space.n)

    with open(args.model, 'rb') as f:
        weights = cPickle.load(f)

    for p, w in zip(params, weights):
        p.set_value(w)

    rewards = []
    for epoch in xrange(args.episodes):
        steps = 0
        reward = 0

        s = env.reset()
        terminal = False
        while not terminal:
            probs = prob_fn([s])[0]
            action = multinomial(probs, args.sample)

            s, r, terminal, _ = env.step(action)
            reward += r
            steps +=1

        print 'epoch {}, steps {}, reward {}'.format(epoch, steps, reward)
        rewards.append(reward)

        if (epoch + 1) % 10 == 0:
            print 'running average reward is {}'.format(sum(rewards)/len(rewards))

    print 'final reward is {}'.format(sum(rewards)/len(rewards))

if __name__ == '__main__':
    test()
