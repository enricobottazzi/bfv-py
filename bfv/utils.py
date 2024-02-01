from bfv.polynomial import Polynomial, get_centered_remainder
    
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
