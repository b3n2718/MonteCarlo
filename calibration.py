import numpy as np
def gbm_parameters(returns):
    parameters = {}
    parameters['sigma'] = np.std(returns[-60:]) * np.sqrt(252)
    parameters['mu'] = np.mean(returns[-60:]) * 252
