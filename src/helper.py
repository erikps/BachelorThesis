from typing import TypeVar, Generic

import torch


ReplayBufferItemType = TypeVar("ReplayBufferItemType")


class ReplayBuffer(Generic[ReplayBufferItemType]):
    """Deque-like tensor-based buffer to store experiences for experience replay."""

    def __init__(self, maxlen: int):
        self._maxlen = maxlen
        self._buffer = torch.empty(maxlen)

        self._len = 0
        self._start_index = 0
        self._end_index = 0

    def _get_index(self, index: int):
        return (self._start_index + index) % self._maxlen

    def __getitem__(self, index: int) -> ReplayBufferItemType:
        if index >= self._len:
            raise IndexError(f"Index {index} out of range, length is {self._len}")
        else:
            return self._buffer[index]

    def __setitem__(self, index: int, item: ReplayBufferItemType):
        if index >= self._len:
            raise IndexError(f"Index {index} out of range, length is {self._len}")
        else:
            return self._buffer[index]

    def __len__(self):
        return self._len

    def append(self, item: ReplayBufferItemType):
        self._len = min(self._maxlen - 1, self._len + 1)
        self[self._end_index] = item
        self._end_index = (self._end_index + 1) % self._maxlen

        if (
            self._len == self._maxlen
        ):  # array is full, need to start moving the start index
            self._start_index = (self._start_index + 1) % self._len

    def sample(self, batch_size: int) -> torch.Tensor[ReplayBufferItemType]:
        """Return 'batch_size'-number of elements randomly from the replay buffer."""

        # pick indices randomly
        indices = torch.randint(len(self), size=(batch_size,))

        # return a list of samples
        return [self[index] for index in indices]
