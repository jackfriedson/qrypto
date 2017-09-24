import itertools
from collections import deque

import numpy as np


class ExperienceBuffer(object):

    def __init__(self, buffer_size, random=None):
        self._buffer_size = buffer_size
        self._total_size = 0
        self._random = random or np.random.RandomState()
        self._buffers = deque()
        self._buffers.append(deque())

    def new_episode(self):
        self._buffers.append(deque())

    def add(self, observation):
        self._buffers[-1].append(observation)
        self._total_size += 1

        if self._total_size > self._buffer_size:
            self._buffers[0].popleft()
            if not self._buffers[0]:
                self._buffers.popleft()

    def sample(self, batch_size, trace_length):
        # TODO: Implement prioritized experience replay
        valid_buffers = list(filter(lambda b: len(b) > trace_length, self._buffers))
        sampled_buffers_idxs = self._random.choice(len(valid_buffers), size=batch_size, replace=True)
        sampled_buffers = np.take(valid_buffers, sampled_buffers_idxs, axis=0)
        sampled_traces = []
        for buf in sampled_buffers:
            start = self._random.randint(0, len(buf) + 1 - trace_length)
            sampled_traces.extend(list(itertools.islice(buf, start, start+trace_length)))
        return sampled_traces
