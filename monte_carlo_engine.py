from payoff import Payoff
import numpy as np
import path_generators

class MonteCarloEngine:
    def __init__(self, model: path_generators.mcmodel, payoff: Payoff, T: float, n_paths: int, n_steps: int, riskfreerate: float):
        self.model = model
        self.payoff = payoff
        self.T = T
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.riskfreerate = riskfreerate

    def price(self, return_std_err=False):
        self.paths = self.model(
            num_paths=self.n_paths, num_steps=self.n_steps,T=self.T
        )
        payoffs = self.payoff(self.paths)
        discounted = np.exp(-self.riskfreerate * self.T) * payoffs
        price = np.mean(discounted)
        stderr = np.std(discounted, ddof=1) / np.sqrt(len(discounted))

        if return_std_err:
            return price, stderr
        return price
    
    def greeks(self):
        
