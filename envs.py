import os
import cv2
import atari_py
import numpy as np
from collections import deque
from gym.spaces.box import Box
from math import sqrt
from gym.envs.atari import AtariEnv
from ale_python_interface import ALEInterface
from game_config import VERSIONS
from gym import error, utils, spaces


class Atari(AtariEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='pong', obs_type='ram', frameskip=(2, 5), repeat_action_probability=0.):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, game, obs_type)
        assert obs_type in ('ram', 'image')

        self.game_path = atari_py.get_game_path(game)
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist'%(game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.ale = ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        self._seed()

        (screen_width, screen_height) = self.ale.getScreenDims()
        self._buffer = np.empty((screen_height, screen_width, 3), dtype=np.uint8)

        self._action_set = self.ale.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width,screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=np.zeros(128), high=np.zeros(128)+255)
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _get_image(self):
        return self.ale.getScreenRGB(self._buffer).copy()


def _process_frame(frame):
    frame = cv2.resize(frame, (84, 84))
    #frame = cv2.resize(frame, (42, 42))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def create_env(env_name, n_frames, version='v0', exploration_bonus=False):
    params = VERSIONS[version]
    env = Atari(env_name, **params)
    return AtariStackFrames(env, n_frames, exploration_bonus)


class AtariStackFrames(object):
    def __init__(self, env, n_frames=4, exploration_bonus=False):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = Box(0, 255, [n_frames, 84, 84])
        self.n_frames = n_frames
        self._checkpoint_buffer = []
        self.buffer = deque(maxlen=n_frames)
        self.exploration_bonus = exploration_bonus
        self.pseudo_counts = np.zeros((42*42, 256), dtype='int')

    def reset(self):
        self.buffer.clear()
        observation = self.env.reset()
        frame = _process_frame(observation)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        for _ in xrange(self.n_frames):
            self.buffer.append(frame)
        return np.array(self.buffer)

    def step(self, action):
        s, r, t, info = self.env.step(action)
        s = _process_frame(s)

        if self.exploration_bonus:
            s_r = cv2.resize(s, (42, 42)).ravel()
            # add observation
            self.pseudo_counts[np.arange(42 * 42), s_r] += 1

            # calculate current density
            n = self.pseudo_counts[np.arange(42 * 42), s_r]
            N = self.pseudo_counts.sum(axis=1)
            p = np.prod(n.astype('float') / N)

            n_after = n + 1
            N_after = N + 1
            p_after = np.prod(n_after.astype('float') / N_after)

            if p_after == p:
                pseudo_cnt = 0
            else:
                pseudo_cnt = p * (1 - p_after) / (p_after - p)
            r_plus = 0.01 / sqrt(pseudo_cnt + 0.01)
        else:
            r_plus = 0

        # process frame
        s = s.astype(np.float32)
        s *= (1.0 / 255.0)
        self.buffer.append(s)

        return np.array(self.buffer), r, t, r_plus

    def load_from_checkpoint(self):
        for frame in self._checkpoint_buffer:
            self.buffer.append(frame)
        del self._checkpoint_buffer[:]
        self.env.ale.loadState()

    def create_checkpoint(self):
        for frame in self.buffer:
            self._checkpoint_buffer.append(frame)
        self.env.ale.saveState()
