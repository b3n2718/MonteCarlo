import numpy as np
from path_generators import *
import pandas as pd

def sample_heston_parameters():
    return {
        'mu': 0.05,
        'V0': np.random.uniform(0.01, 0.2),
        'kappa': np.random.uniform(0.5, 5.0),
        'theta': np.random.uniform(0.01, 0.2),
        'xi': np.random.uniform(0.1, 1.0),
        'rho': np.random.uniform(-0.9, 0.0)
    }

def prepare_dataset(model_class, num_param_sets=5000, paths_per_param=125,
                    num_steps=252, T=1.0, S0=100.0, normalize=True):
    """
    Baut ein Dataset aus Heston-Simulationen
    Returns:
        X: np.ndarray, shape (num_param_sets * paths_per_param, num_steps)
        y: np.ndarray, shape (num_param_sets * paths_per_param, 5)
    """
    X = []
    y = []
    
    for _ in range(num_param_sets):
        params = sample_heston_parameters()
        model = model_class(parameters={**params, 'mu': params['mu']})
        paths = model(num_paths=paths_per_param, num_steps=num_steps, T=T, S0=S0)  # shape (paths, steps)
        
        # Optional: auf Returns umstellen oder normalisieren
        if normalize:
            paths = np.log(paths[:, 1:] / paths[:, :-1])  # Log-Returns
        
        # Daten und Labels speichern
        for i in range(paths.shape[0]):
            X.append(paths[i])
            y.append([params['kappa'], params['theta'], params['xi'], params['rho'], params['V0']])
    
    return np.array(X), np.array(y)


def save_heston_dataset_hdf5(X: np.ndarray, y: np.ndarray, filename: str = "heston_dataset.h5"):
    """
    Speichert das Dataset als HDF5-Datei über pandas.
    X: np.ndarray, shape (n_samples, n_timesteps)
    y: np.ndarray, shape (n_samples, 5)
    """
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y, columns=["kappa", "theta", "xi", "rho", "V0"])

    with pd.HDFStore(filename, mode='w') as store:
        store.put("X", df_X)
        store.put("y", df_y)
    
    print(f"✅ Dataset gespeichert als '{filename}'")

if __name__=='__main__':
    X,Y =prepare_dataset(heston)
    save_heston_dataset_hdf5(X,Y)