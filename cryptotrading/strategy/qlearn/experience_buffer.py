from collections import deque


class ExperienceBuffer(object):

    def __init__(self, buffer_size):
        self._buffer = deque(maxlen=buffer_size)

    def add(self, observation):
        self._buffer.append(observation)

    def sample(self, batch_size, trace_length):
        samples = random.sample(self._buffer, batch_size)
        