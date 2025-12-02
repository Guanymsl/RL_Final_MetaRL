import numpy as np

def standardize(arr: np.ndarray) -> np.ndarray:
    eps = 1e-8
    mu = arr.mean()
    sigma = arr.std()
    standardized = (arr - mu) / (eps + sigma)
    return standardized

def explained_variance(ypred: np.ndarray, y: np.ndarray) -> np.float32:
    vary = y.var()
    return 1 - (y-ypred).var()/(1e-8 + vary)
