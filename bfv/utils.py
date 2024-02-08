from bfv.polynomial import Polynomial, get_centered_remainder
from typing import List
import math

def mod_inverse_centered(t, q):
    """
    Computes the multiplicative inverse of t modulo q expressed in the centered remainder representation.
    Returns the inverse, or raises an exception if it doesn't exist.
    """
    g, x, _ = extended_gcd(t, q)
    if g != 1:
        raise ValueError("The multiplicative inverse does not exist")
    else:
        return get_centered_remainder(x, q)

def mod_inverse(t, q):
    """
    Computes the multiplicative inverse of t modulo q.
    Returns the inverse, or raises an exception if it doesn't exist.
    """
    g, x, _ = extended_gcd(t, q)
    if g != 1:
        raise ValueError("The multiplicative inverse does not exist")
    else:
        return x % q
    
def extended_gcd(a, b):
    """
    Computes the greatest common divisor of a and b.
    Returns a tuple (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)
    
def adjust_negative_coefficients(poly : Polynomial, modulus: int) -> Polynomial:
    """
    Adjust the coefficients of the polynomial to be positive.
    """
    return Polynomial([(modulus + coeff if coeff < 0 else coeff) for coeff in poly.coefficients])

def are_coprime(a, b):
    """Check if a and b are coprime, i.e., gcd(a, b) == 1."""
    return math.gcd(a, b) == 1

def find_odd_pairwise_coprimes(base: int, count: int, max_step=1000) -> List[int]:
    """
    Generate a list of `count` odd coprime numbers starting from `base`.
    
    This function ensures that all numbers in the list are odd and coprime with each other.
    It starts from the next odd number greater than or equal to `base` and increments the candidate by 2
    to only consider odd candidates.
    """
    # Ensure starting base is odd
    if base % 2 == 0:
        base += 1

    coprimes = [base]
    candidate = base + 2  # Start from the next odd number after base
    
    while len(coprimes) < count:
        if all(are_coprime(candidate, coprime) for coprime in coprimes):
            coprimes.append(candidate)
        candidate += 2  # Increment by 2 to ensure candidate is always odd
        
        if candidate - base > max_step:  # Prevent infinite loop in case count is too high
            break
    assert len(coprimes) == count, "Failed to find enough coprime numbers"
    return coprimes
