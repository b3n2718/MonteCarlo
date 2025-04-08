import monte_carlo_engine
import numpy as np
import payoff
import path_generators
# Parameter
num_paths = 50000
num_steps = int(252)
S0 = 100
rr = 0.04
sigma = 0.1
T=1
X=120

engine = monte_carlo_engine.MonteCarloEngine(path_generators.gbm({'mu':rr,'sigma':sigma}), payoff.EuropeanCall(X),T,num_paths,num_steps,rr,S0)
print(engine.price())
print(engine.greeks())
print(engine.price())


#plt.plot(engine.paths.T)
#plt.show()




