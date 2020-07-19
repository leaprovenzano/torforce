from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from torforce.ops import expanded_empty_like


@dataclass
class Memory:

    step: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor

    @classmethod
    def fields(cls):
        yield from cls.__dataclass_fields__.keys()

    def items(self):
        for k in self.fields():
            yield (k, getattr(self, k))

    def __setitem__(self, i, v):
        if isinstance(v, self.__class__):
            for k in self.fields():
                getattr(self, k)[i] = getattr(v, k)
        return NotImplemented

    def __getitem__(self, i):
        return self.__class__(**{k: v[i] for k, v in self.items()})


class FixedStepMemory(Dataset):
    def __init__(self, steps: int, n_workers: int = 1):
        self.n_workers = n_workers
        self._memory = None
        self.steps = steps
        self._step = 0
        self._flat_indices = torch.arange(len(self))

    def __len__(self):
        return self.steps * self.n_workers

    def _init_memory(self, step_memory):
        memory_type = step_memory.__class__
        self._memory = memory_type(
            **{k: expanded_empty_like(v, self.steps, ...) for k, v in step_memory.items()}
        )

    def update(self, step_memory):
        if self._memory is None:
            self._init_memory(step_memory)
        self._memory[self._step] = step_memory
        self._step += 1

    def __repr__(self):
        return self._memory.__repr__()

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            ix = self._flat_indices[ix]
        i, j = ix % self.steps, ix // self.steps
        return self._memory[i, j]

    def __getattr__(self, k):
        return getattr(self._memory, k)
