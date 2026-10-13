import numpy as np
from inspy.energy import Energy
from inspy.instrument import get_tau, TripleAxisSpectr


# --- minimal instrument setup ---
tas = TripleAxisSpectr(efixed=14.7, method=1)  # Popovici

# --- simple Gaussian sqw: one mode, returns (1, N) ---
def sqw_gauss(H, K, L, W, p):
    sigma = p[0]
    W0    = p[1]
    return np.exp(-0.5 * ((W - W0) / sigma) ** 2)[np.newaxis, :]  # (1, N)

p = [1.0, 5.0]  # sigma=1 meV, peak at 5 meV

# --- single Q point ---
hkle = ([1.0], [0.0], [0.0], [5.0])

# --- reference: fix method ---
conv_fix = tas.ResConv(sqw_gauss, None, 0, hkle,
                       METHOD='fix', ACCURACY=[7, 3], p=p)

# --- test: mc method ---
conv_mc = tas.ResConv(sqw_gauss, None, 0, hkle,
                      METHOD='mc', ACCURACY=50, p=p, seed=42)

print(f"fix result : {conv_fix[0]:.6f}")
print(f"mc  result : {conv_mc[0]:.6f}")
print(f"relative diff: {abs(conv_fix[0] - conv_mc[0]) / conv_fix[0] * 100:.2f}%")