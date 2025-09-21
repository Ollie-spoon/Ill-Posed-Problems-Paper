from jax import numpy as jnp
from matplotlib import pyplot as plt

# Create a time array
n = 2726
dt = 480 # microseconds used to avoid floating point errors in t
t_max = n * dt
t = jnp.linspace(0, t_max, n, endpoint=False)/1000 # convert to milliseconds

# Create a clean signal
params = jnp.array([
    0.5, 40.0,
    0.5, 200.0,
])
signal = params[0] * jnp.exp(-t/params[1]) + params[2] * jnp.exp(-t/params[3])

# Iterate through the derivatives of the signal until a non-physical value is found
for i in range(10):
    
    # Take the derivative of the signal
    derivative = signal if i == 0 else jnp.gradient(derivative, dt/1000)[1:]
    
    # Calculate the appropriate sign for the derivative
    derivative_sign = (-1)**(i)
    print(f"derivative {i} sign: {derivative_sign}")
    
    # find if there are any values that have the wrong sign
    wrong_sign = jnp.where(derivative * derivative_sign < 0, 1, 0)
    
    wrong_count = jnp.sum(wrong_sign)
    if wrong_count == 0:
        print(f"Derivative {i} has no non-physical values\n")
    else:
        print(f"Derivative {i} has {wrong_count}/{len(derivative)} non-physical values")
        break

# plot the derivative
plt.figure(figsize=(7, 3))
plt.axhline(0.0, color='black', linestyle='--', linewidth=1)
plt.plot(derivative, label=f'Derivative {i}', color='orange', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Signal Derivative Amplitude')
plt.xticks([])
plt.yticks([])
plt.legend()
plt.show()
