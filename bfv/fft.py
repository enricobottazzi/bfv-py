from typing import List
from mpmath import *
import os
dps = os.getenv('DPS')
mp.dps = int(dps)

def recursive_fft(a: List[int]) -> List[complex]:
    """
    Compute the recursive FFT of a list of integers
    """
    n = len(a)
    if n == 1:
        return a
    else:
        i = 1j
        w_n = e ** (2 * i * pi / float(n))
        w = 1
        a_even = [a[i] for i in range(0, n, 2)]
        a_odd = [a[i] for i in range(1, n, 2)]
        y_even = recursive_fft(a_even)
        y_odd = recursive_fft(a_odd)
        y = [complex(0, 0)] * n
        for k in range(n // 2):
            y[k] = y_even[k] + w * y_odd[k]
            y[k + n // 2] = y_even[k] - w * y_odd[k]
            w = w * w_n
        return y
    
def unscaled_recursive_fft(y: List[complex]) -> List[complex]:
    n = len(y)
    if n == 1:
        return y
    else:
        i = 1j
        w_n = e ** (-2 * i * pi / float(n))
        w = 1
        y_even = [y[i] for i in range(0, n, 2)]
        y_odd = [y[i] for i in range(1, n, 2)]
        a_even = unscaled_recursive_fft(y_even)
        a_odd = unscaled_recursive_fft(y_odd)
        a = [complex(0, 0)] * n
        for k in range(n // 2):
            a[k] = a_even[k] + w * a_odd[k]
            a[k + n // 2] = a_even[k] - w * a_odd[k]
            w = w * w_n
        return a

def recursive_ifft(y: List[complex]) -> List[complex]:
    """
    Compute the recursive IFFT of a list of complex numbers
    Check https://stackoverflow.com/questions/48572647/recursive-inverse-fft 
    """
    coeffs = unscaled_recursive_fft(y)
    coeffs = [coeff / len(coeffs) for coeff in coeffs]     
    coeffs = [nint(coeff.real) for coeff in coeffs]  
    return coeffs

def find_n_th_roots_of_unity(n: int) -> List[complex]:
    """
    Find the n-th roots of unity.
    """
    roots = [exp(2j * pi * i / n) for i in range(n)]

    return roots