import numpy as np
from abc import ABC, abstractmethod

class Payoff(ABC):
    @abstractmethod
    def __call__(self, paths: np.ndarray) -> np.ndarray:
        pass

class European(Payoff):
    def __init__(self, strike, side):
        self.strike = strike
        if side in ['put','call']:
            self.side = side
        else:
            raise ValueError("side must be put or call")

    def __call__(self, paths):
        if self.side == "call":
            return np.maximum(paths[:, -1] - self.strike, 0.0)
        elif self.side == "put":
            return np.maximum(self.strike - paths[:, -1], 0.0)

class Barrier(Payoff):
    def __init__(self, strike, side, barrier, barrier_type='up-and-out'):
        self.strike = strike
        if side in ['put','call']:
            self.side = side
        else:
            raise ValueError("side must be put or call")

        self.barrier = barrier
        self.barrier_type = barrier_type

    def __call__(self, paths):
        if self.barrier_type == 'up-and-out':
            knocked_out = np.any(paths >= self.barrier, axis=1)
            if self.side == "call":
                payoff = np.maximum(paths[:, -1] - self.strike, 0.0)
            elif self.side == "put":
                payoff = np.maximum(self.strike - paths[:, -1], 0.0)
            
            payoff[knocked_out] = 0.0
            return payoff
        else:
            raise NotImplementedError("Nur up-and-out bisher implementiert")

class Asian(Payoff):
    def __init__(self, strike, side):
        self.strike = strike
        if side in ['put','call']:
            self.side = side
            
    def __call__(self, paths):
        avg = np.mean(paths, axis=1)
        if self.side == "call":
            return np.maximum(avg - self.strike, 0.0)
        if self.side == "put":
            return np.maximum(self.strike - avg, 0.0)
