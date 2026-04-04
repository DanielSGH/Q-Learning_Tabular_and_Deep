

from typing import List, Tuple
import numpy as np


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.storage: List[Transition] = []
        self.next_idx = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition: Transition = (
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
        )

        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.next_idx] = transition

        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size > len(self.storage):
            raise ValueError("batch_size cannot exceed current buffer size")

        idxs = np.random.choice(
            len(self.storage), size=batch_size, replace=False)
        batch = [self.storage[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)

        states_arr = np.stack(states, axis=0).astype(np.float32, copy=False)
        actions_arr = np.asarray(actions, dtype=np.int64)
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        next_states_arr = np.stack(
            next_states, axis=0).astype(np.float32, copy=False)
        dones_arr = np.asarray(dones, dtype=np.float32)

        return states_arr, actions_arr, rewards_arr, next_states_arr, dones_arr

    def __len__(self) -> int:
        return len(self.storage)
