from .engine import monte_carlo
import numpy as np
from abc import ABC, abstractmethod
import concurrent.futures
import multiprocessing as mp

class MCPathGenerator():
    def __init__(self):
        pass
    
    @abstractmethod
    def generate_paths(self) -> np.ndarray:
        pass
    
    def update_params(self,params):
        self.__dict__.update(params)

def simulate_chunk(num_paths,num_steps,S0, mu, sigma, dt):
    return np.array(monte_carlo.gbm(num_paths,num_steps,S0, mu, sigma, dt))  

class GBMPathGenerator(MCPathGenerator):
    def __init__(self, parameters: dict):
        """
        Init function for gbm model
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



    def __call__(self, num_paths: int, num_steps: int, T: float,S0: float, parallel=False, n_processes=None):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        if parallel:
            if n_processes is None:
                n_processes = mp.cpu_count()
                
            chunk_size = num_paths // n_processes
            remainder = num_paths % n_processes

            chunk_sizes = [chunk_size] * n_processes
            for i in range(remainder):
                chunk_sizes[i] += 1  # verteile die restlichen Pfade   
            args = [(chunk_sizes[i], self.num_steps,S0, self.mu, self.sigma, self.dt) for i in range(n_processes)]
            
          
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(simulate_chunk, *arg) for arg in args]
                results = [f.result() for f in futures]
                    

            return np.vstack(results)
        else:
            return np.array(monte_carlo.gbm(self.num_paths,self.num_steps,S0, self.mu, self.sigma, self.dt))

class JumpDiffusionPathGenerator(MCPathGenerator):
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

  
    def __call__(self, num_paths: int, num_steps: int, T: float,S0: float):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        return np.array(monte_carlo.jump_diffusion(self.num_paths,self.num_steps, S0, self.mu, 
                                                   self.sigma, self.mu_j, self.sigma_j, self._lambda, self.dt))

class HestonPathGenerator(MCPathGenerator):
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

class BatesPathGenerator(MCPathGenerator):
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

        
    def __call__(self, num_paths: int, num_steps: int, T: float, S0: float):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        return np.array(monte_carlo.bates(self.num_paths, self.num_steps, S0, self.V0, self.mu, 
                        self.kappa, self.theta, self.xi, self.rho, self.mu_j, self.sigma_j, self._lambda, self.dt))

class VarainceGammaPathGenerator(MCPathGenerator):
    def __init__(self, parameters: dict):
        """
        Init function for VarainceGamma model
        Args:
            num_paths (int): Number of paths
            num_steps (int): NUmber of time steps
            T (float): Total duration
            parameters (dict): 'mu','sigma','gamma','alpha','beta', are required
        """
        required_params = {'mu','sigma','gamma','alpha','beta'}
        if not required_params.issubset(parameters.keys()):
            raise ValueError(f"Parameter list incomplete for model expected {required_params}")
        
        super().__init__()
        self.__dict__.update(parameters)
 

    def __call__(self, num_paths: int, num_steps: int, T: float, S0: float):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        return np.array(monte_carlo.variance_gamma(self.num_paths, self.num_steps, S0, self.mu, self.sigma, self.gamma, self.alpha, self.beta, self.dt))

class VasicekPathGenerator(MCPathGenerator):
    def __init__(self, parameters: dict):
        """
        Init function for Vasicek model
        Args:
            num_paths (int): Number of paths
            num_steps (int): NUmber of time steps
            T (float): Total duration
            parameters (dict): 'theta','sigma','kappa' are required
        """
        required_params = {'theta','sigma','kappa'}
        if not required_params.issubset(parameters.keys()):
            raise ValueError(f"Parameter list incomplete for model expected {required_params}")
        
        super().__init__()
        self.__dict__.update(parameters)
 

    def __call__(self, num_paths: int, num_steps: int, T: float, r0: float):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        return np.array(monte_carlo.vasicek(self.num_paths, self.num_steps, r0, self.theta, self.sigma, self.kappa, self.dt))

class CIRPathGenerator(MCPathGenerator):
    def __init__(self, parameters: dict):
        """
        Init function for Cox-Ingersoll-Ross model
        Args:
            num_paths (int): Number of paths
            num_steps (int): NUmber of time steps
            T (float): Total duration
            parameters (dict): 'theta','sigma','kappa' are required
        """
        required_params = {'theta','sigma','kappa'}
        if not required_params.issubset(parameters.keys()):
            raise ValueError(f"Parameter list incomplete for model expected {required_params}")
        
        super().__init__()
        self.__dict__.update(parameters)
 

    def __call__(self, num_paths: int, num_steps: int, T: float, r0: float):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T/num_steps
        return np.array(monte_carlo.cir(self.num_paths, self.num_steps, r0, self.theta, self.sigma, self.kappa, self.dt))