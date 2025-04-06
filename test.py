import monte_carlo_engine
import numpy as np
import payoff
import path_generators
# Parameter
num_paths = 1000
num_steps = int(252)
S0 = 5000
rr = 0.04
sigma = 0.4
T=2/12
X=5300

engine = monte_carlo_engine.MonteCarloEngine(path_generators.gbm({'S0':S0,'mu':rr,'sigma':sigma}), payoff.EuropeanCall(X),T,num_paths,num_steps,rr)
print(engine.price())

import matplotlib.pyplot as plt

plt.plot(engine.paths.T)
plt.show()




