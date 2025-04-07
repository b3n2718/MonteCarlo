from payoff import Payoff
import numpy as np
import path_generators
import copy

class MonteCarloEngine:
    def __init__(self, model: path_generators.mcmodel, payoff: Payoff, T: float, n_paths: int, n_steps: int, riskfreerate: float, S0:float):
        self.model = model
        self.payoff = payoff
        self.T = T
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.riskfreerate = riskfreerate
        self.S0 = S0

    def price(self,S0=None, return_std_err=False):
        if S0 is None:
            S0 = self.S0
        self.paths = self.model(
            num_paths=self.n_paths, num_steps=self.n_steps,T=self.T,S0=S0
        )
        payoffs = self.payoff(self.paths)
        discounted = np.exp(-self.riskfreerate * self.T) * payoffs
        price = np.mean(discounted)
        stderr = np.std(discounted, ddof=1) / np.sqrt(len(discounted))

        if return_std_err:
            return price, stderr
        return price
    
    def __price(self,S0,model):
        if S0 is None:
            S0 = self.S0
        self.paths = model(
            num_paths=self.n_paths, num_steps=self.n_steps,T=self.T,S0=S0
        )
        payoffs = self.payoff(self.paths)
        discounted = np.exp(-self.riskfreerate * self.T) * payoffs
        price = np.mean(discounted)
        stderr = np.std(discounted, ddof=1) / np.sqrt(len(discounted))

        if return_std_err:
            return price, stderr
        return price        
    
    def greeks(self):
        model_param_copy=copy.deepcopy(self.model.__dict__)
        greeks = {}
        
        #delta
        dS=0.01*self.S0
        greeks['delta'] = (self.price(self.S0+dS)-self.price(self.S0-dS))/(2*dS)
        #vega
        #dsigma=model_param_copy['sigma']/0.01
        #self.model.__dict__['sigma'] += dsigma
        #vp=self.price()
        #self.model.__dict__.update(model_param_copy)
        #self.model.__dict__['sigma'] -= dsigma
        #vm=self.price()
        #self.model.__dict__.update(model_param_copy)
        #greeks['vega'] = (vp-vm)/(2*dsigma)
        

        
        
        return greeks