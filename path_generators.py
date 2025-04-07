from engine import monte_carlo
import numpy as np
from abc import ABC, abstractmethod

class mcmodel():
    def __init__(self):
        pass
    
    @abstractmethod
    def generate_paths(self) -> np.ndarray:
        pass

class gbm(mcmodel):
    def __init__(self, parameters: dict):
        """
        Init function for gmb model
        Args:
            num_paths (int): Number of paths
            num_steps (int): NUmber of time steps
            T (float): Total duration
            parameters (dict): mu, sigma are required
        """
        required_params = {'mu','sigma'}
        if not required_params.issubset(parameters.keys()):
            raise ValueError(f"Parameter list incomplete for model expected {required_params}")
        
        super().__init__()
        self.__dict__.update(parameters)


    def __call__(self, num_paths: int, num_steps: int, T: float,S0: float):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        return np.array(monte_carlo.gbm(self.num_paths,self.num_steps,S0, self.mu, self.sigma, self.dt))

class jump_diffusion(mcmodel):
    def __init__(self, parameters: dict):
        """
        Init function for jump_diffusion model
        Args:
            num_paths (int): Number of paths
            num_steps (int): NUmber of time steps
            T (float): Total duration
            parameters (dict): mu, sigma are required
        """
        required_params = {'mu','sigma','mu_j','sigma_j','_lambda'}
        if not required_params.issubset(parameters.keys()):
            raise ValueError(f"Parameter list incomplete for model expected {required_params}")
        
        super().__init__()
        self.__dict__.update(parameters)

  
    def __call__(self, num_paths: int, num_steps: int, T: float,):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        return np.array(monte_carlo.jump_diffusion(self.num_paths,self.num_steps, self.S0, self.mu, 
                                                   self.sigma, self.mu_j, self.sigma_j, self._lambda, self.dt))

class heston(mcmodel):
    def __init__(self, parameters: dict):
        """
        Init function for heston model
        Args:
            num_paths (int): Number of paths
            num_steps (int): NUmber of time steps
            T (float): Total duration
            parameters (dict): 'mu',,'V0','kappa','theta','xi','rho' are required
        """
        required_params = {'mu','V0','kappa','theta','xi','rho'}
        if not required_params.issubset(parameters.keys()):
            raise ValueError(f"Parameter list incomplete for model expected {required_params}")
        
        super().__init__()
        self.__dict__.update(parameters)


    def __call__(self, num_paths: int, num_steps: int, T: float,S0):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        return np.array(monte_carlo.heston(self.num_paths, self.num_steps, S0, self.V0, self.mu, 
                                           self.kappa, self.theta, self.xi, self.rho, self.dt))

class bates(mcmodel):
    def __init__(self, parameters: dict):
        """
        Init function for heston model
        Args:
            num_paths (int): Number of paths
            num_steps (int): NUmber of time steps
            T (float): Total duration
            parameters (dict): 'mu','sigma','V0','kappa','theta','xi','rho','mu_j','sigma_j','_lambda' are required
        """
        required_params = {'mu','sigma','V0','kappa','theta','xi','rho','mu_j','sigma_j','_lambda'}
        if not required_params.issubset(parameters.keys()):
            raise ValueError(f"Parameter list incomplete for model expected {required_params}")
        
        super().__init__()
        self.__dict__.update(parameters)

        
    def __call__(self, num_paths: int, num_steps: int, T: float):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        return np.array(self.num_paths, self.num_steps, self.S0, self.V0, self.mu, 
                        self.kappa, self.theta, self.xi, self.rho, self.mu_j, self.sigma_j, self._lambda, self.dt)

def varaince_gamma():
    pass
    

