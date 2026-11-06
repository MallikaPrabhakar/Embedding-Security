
#* This file contains code for FOWH algorithm with theoretical error correction + utility guarantees, without security guarantees. 

from sage.all import *
import numpy as np
from time import time
from decimal import *
getcontext().prec = 50

#* Preprocessing functions
seed  = 0
set_random_seed(seed)

# Convert real vector into integer vector mod q
# max_proj=6.1, min_proj=-5.3 # for FEI projected and eigen (inverse) scaled dataset: min val: -5.203783132056765 max val: 6.003189324296841 mean val: -0.0018198492298009629 var val: 1.2554071156231181
def zq_x(x, q, max_proj, min_proj, scale):
    # almost all entries of x are from min_proj to max_proj
    x -= min_proj
    x /= (max_proj - min_proj)
    x *= q
    x = np.floor(x).astype(int) % q 
    if scale:
        x = x*scale 
        x = np.floor(x).astype(int) % q
    return x

def process_input(x, q, max_proj, min_proj, scale): # use q = None to not change ring to Zq 
    x = x.copy()
    if isinstance(x, np.ndarray):
        x = zq_x(x, q, max_proj, min_proj, scale)
        x = vector(x)  # Convert to Sage vector
        # print(x)
        if q:
            x = x.change_ring(GF(q))
            # print(x)
    return x

def fast_lll(B, delta=None, verbose=False):
    if delta is None:
            delta = ZZ(99)/ZZ(100)
    if verbose:
        print("Creating lattice and LLL reduce", flush=True)
    start = time()
    B = matrix([v for v in B.LLL(delta=delta) if v])
    end = time()
    if verbose:
        print(f"Time taken for lll reduce: {(end - start) // 60} minutes ({int(end - start)} seconds)", flush=True)
    return B