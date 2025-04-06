import numpy as np
from abc import ABC, abstractmethod

class Payoff(ABC):
    @abstractmethod
    def __call__(self, paths: np.ndarray) -> np.ndarray:
        pass

class EuropeanCall(Payoff):
    def __init__(self, strike):
        self.strike = strike

    def __call__(self, paths):
        return np.maximum(paths[:, -1] - self.strike, 0.0)

class BarrierCall(Payoff):
    def __init__(self, strike, barrier, barrier_type='up-and-out'):
        self.strike = strike
        self.barrier = barrier
        self.barrier_type = barrier_type

    def __call__(self, paths):
        if self.barrier_type == 'up-and-out':
            knocked_out = np.any(paths >= self.barrier, axis=1)
            payoff = np.maximum(paths[:, -1] - self.strike, 0.0)
            payoff[knocked_out] = 0.0
            return payoff
        else:
            raise NotImplementedError("Nur up-and-out bisher implementiert")

class AsianCall(Payoff):
    def __init__(self, strike):
        self.strike = strike

    def __call__(self, paths):
        avg = np.mean(paths, axis=1)
        return np.maximum(avg - self.strike, 0.0)
