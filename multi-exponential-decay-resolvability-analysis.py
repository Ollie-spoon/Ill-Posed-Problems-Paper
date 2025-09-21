from jax import numpy as jnp
from jax import lax, jit

def domain_limited_resolution(y: float, snr: float, N: int = 100):
    """
    Calculate the domain-limited resolution for a given range of decay constants.
    
    This analysis relies on a couple of key pieces of information:
    The reformulation of the Laplace transform into
    
        (K*Kf)(t) = \int_a^b 1/(t+q) f(s)ds
    
    where K is the kernel operator and f(s) is the function we are trying to fit.
    
    The eigenvalues of the kernel operator K*K are related to the singular values of K.
    
    We approximate the eigenvalue problem on the left hand side using the composite 
    trapezium rule applied to the right hand side.
    
    The singular values of K are then recovered using the square roots of the eigenvalues 
    of the kernel operator K*K.
    
    The Ks value, the number of resolvable decays, is determined by the largest index k 
    such that a_k >= 1/SNR, where a_k is the k-th singular value. This is rarely an integer,
    so we can linearly interpolate between the two relevant singular values to get a 
    non-integer Ks.
    
    This Ks is then used to calculate the resolution ratio delta_s, which is given by:
    
        delta_s = (b/a)^(1/Ks)
    
    Parameters:
        y: The range of decay constants (b/a) we are interested in.
        snr: The signal-to-noise ratio (SNR).
        N: The number of discretization points for the integration.
    
    Returns:
        delta_s: The resolution ratio.
    
    """
    
    # 1) discretize & build weighted matrix
    a, b = 1.0, y
    thresh = 1.0 / snr
    
    # discretize the interval [a, b]
    t = jnp.linspace(a, b, N)
    
    # uniform spacing over [a,b]
    h = (b - a) / (N - 1)
    weights = jnp.ones(N)*h
    
    # edge terms are only terms not used twice in trapesium rule so they are halved
    weights = weights.at[0].set(h/2)
    weights = weights.at[-1].set(h/2)
    
    # create the integral matrix that approximates (K*K)
    # This is a standard approach to discretizing the integral operator
    # using the trapezoidal rule
    A = 1.0 / (t[:, None] + t[None, :])
    W = jnp.sqrt(weights)
    Aw = W[:, None] * A * W[None, :]
    
    # 2) eigen to descending sorted singular values 
    eigs, _ = jnp.linalg.eigh(Aw)
    svals = jnp.sqrt(jnp.maximum(jnp.sort(eigs)[::-1], 0.0))
    
    # 3) compute non-integer Ks by interpolation
    @jit
    def compute_Ks(s):

        # count how many pass, minus 1 (index of last valid)
        k0 = jnp.sum(s >= thresh) - 1
        
        # clip to [0, len-2] so we can safely index k0, k0+1
        k0 = jnp.clip(k0, 0, s.shape[0] - 2)
        a0, a1 = s[k0], s[k0 + 1]
        
        # fractional part
        delta = jnp.where(a0 != a1, (a0 - thresh) / (a0 - a1), 0.0)
        return k0 + delta

    # compute Ks using the singular values
    Ks = lax.cond(
        jnp.any(svals >= thresh),
        compute_Ks,          # if True: interpolate
        lambda s: 1e-308,   # if False: 0
        operand=svals
    )
    
    # 4) final resolution ratio and Ks
    return y ** (1.0 / Ks), Ks

# define variables 
a = 5       # minimum decay rate
b = 1000    # maximum decay rate
snr = 100   # signal to noise ratio 

# calculate Ds and Ks through numerical integration scheme
Ds, Ks = domain_limited_resolution(b/a, snr, N=2000)
print(f"With b/a={b/a}, snr={snr} \n\tDs={Ds}\n\tKs={Ks}\n")

# transform values to log-scale for probability calculation
L = jnp.log(b/a)
s = jnp.log(Ds)

# p defines the probability that n values uniformly generated
# over [0, L] have a minimum seperation of at least s
p = lambda L, n, s: (1-(n-1)*s/L)**n

# Iterate through n values from 1 to round_up(Ks)
for n in range(1, int(jnp.ceil(Ks))+1):
    print(f"for n={n}, p={p(L, n, s)}")
