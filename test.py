import monte_carlo
import numpy as np
import time as t
import random as rd
# Parameter
num_paths = 10000
num_steps = 252
S0 = 100
mu = 0.05
sigma = 0.2
dt = 1/252

# Simulation starten
def mc_gbm(num_paths, num_steps, S0, mu, sigma, dt):
    r=np.zeros((num_paths,num_steps))
    for i in range(num_paths):
        S = S0
        for j in range(num_steps):
            S *= np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * rd.gauss(0,1))
            r[i, j] = S
    return r

def mc_gbm_opt(num_paths, num_steps, S0, mu, sigma, dt):
    
    Z = np.random.normal(0, 1, (num_steps, num_paths))  # Zufallszahlen für jeden Zeitschritt
    increments = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)  # Diskretisierte SDE
    log_paths = np.cumsum(increments, axis=0)  # Kumulierte Summe für log(S)
    
    paths = S0 * np.exp(log_paths)  # Umwandlung zu S(t)
    paths = np.vstack([np.full((1, num_paths), S0), paths])  # Startwerte hinzufügen
    
    return paths  # Matrix der simulierten Pfade

t0=t.time()
sim_data = monte_carlo.gbm(num_paths, num_steps, S0, mu, sigma, dt)
print(t.time()-t0)


t0=t.time()
sim_data = mc_gbm_opt(num_paths, num_steps, S0, mu, sigma, dt)
print(t.time()-t0)

t0=t.time()
sim_data = monte_carlo.jump_diffusion(num_paths, num_steps, S0, mu, sigma,1,1,0.5, dt)
print(t.time()-t0)

