from .payoff import Payoff
import numpy as np
from .path_generators import *
import copy

class MonteCarloEngine:
    def __init__(self, model: MCPathGenerator, payoff: Payoff, T: float, n_paths: int, n_steps: int, riskfreerate: float, S0:float):
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
    
    def greeks(self):
        base_model_param_copy=copy.deepcopy(self.model.__dict__)
        model_param_copy=copy.deepcopy(self.model.__dict__)
        greeks = {}
        
        #delta
        dS=0.01*self.S0        
        greeks['delta'] = (self.price(self.S0+dS)-self.price(self.S0-dS))/(2*dS)
        
        #gamma
        greeks['gamma'] = (self.price(self.S0+dS)+self.price(self.S0-dS)-2*self.price(self.S0))/(dS*dS)
        
        #vega
        if isinstance(self.model,GBMPathGenerator) or isinstance(self.model,JumpDiffusionPathGenerator):
            dsigma=base_model_param_copy['sigma']*0.01
            model_param_copy['sigma'] = base_model_param_copy['sigma'] + dsigma
            self.model.update_params(model_param_copy)
            vp=self.price()
            model_param_copy['sigma'] = base_model_param_copy['sigma'] - dsigma
            self.model.update_params(model_param_copy)
            vm=self.price()

            greeks['vega'] = (vp-vm)/(2*dsigma)
        elif isinstance(self.model,HestonPathGenerator) or isinstance(self.model,BatesPathGenerator):
            dsigma=base_model_param_copy['V0']*0.01
            model_param_copy['V0'] = base_model_param_copy['V0'] + dsigma
            self.model.update_params(model_param_copy)
            vp=self.price()
            model_param_copy['V0'] = base_model_param_copy['V0'] - dsigma
            self.model.update_params(model_param_copy)
            vm=self.price()

            greeks['vega'] = (vp-vm)/(2*dsigma)
        self.model.update_params(base_model_param_copy)
        
        #Theta
        T = self.T
        dt=0.01
        self.T = T+dt
        vp=self.price()
        self.T = T-dt
        vm=self.price()
        greeks['theta'] = (vp-vm)/(2*dt)
        self.T = T
        
        #rho
        riskfreerate = self.riskfreerate
        model_param_copy=copy.deepcopy(self.model.__dict__)
        drr=0.001
        self.riskfreerate = riskfreerate + drr
        model_param_copy['mu'] = base_model_param_copy['mu'] + drr
        self.model.update_params(model_param_copy)
        vp=self.price()
        self.riskfreerate = riskfreerate - drr
        model_param_copy['mu'] = base_model_param_copy['mu'] - drr
        self.model.update_params(model_param_copy)
        vm=self.price()
        greeks['rho'] = (vp-vm)/(2*drr)
        self.model.update_params(base_model_param_copy)
        self.riskfreerate = riskfreerate        
        
        return greeks