# Find the arc length of a curve defined by the function y = A * exp(B * x)
# by ChatGPT

import numpy as np
from scipy.integrate import quad

def arc_length_exponential(x1, x2, A, B):
    # Define the integrand for the arc length formula
    integrand = lambda x: np.sqrt(1 + (A * B * np.exp(B * x))**2)
    
    # Perform numerical integration from x1 to x2
    length, error = quad(integrand, x1, x2)
    
    return length

# Example usage:
x1 = 0
x2 = 5
A = 10
B = 0.05

length = arc_length_exponential(x1, x2, A, B)
print(f"Arc length of the curve from x={x1} to x={x2} is approximately {length:.6f}")
