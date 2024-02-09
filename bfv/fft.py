from typing import List
import cmath
from math import pi, e

def recursive_fft(a: List[int]) -> List[complex]:
    """
    Compute the recursive FFT of a list of integers
    """
    n = len(a)
    if n == 1:
        return a
    else:
        i = 1j
        w_n = e ** (-2 * i * pi / float(n))
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
    
def recursive_ifft(y: List[complex]) -> List[complex]:
    """
    Compute the recursive IFFT of a list of complex numbers
    """
    n = len(y)
    if n == 1:
        return y
    else:
        i = 1j
        w_n = e ** (2 * i * pi / float(n))
        w = 1
        y_even = [y[i] for i in range(0, n, 2)]
        y_odd = [y[i] for i in range(1, n, 2)]
        a_even = recursive_ifft(y_even)
        a_odd = recursive_ifft(y_odd)
        a = [complex(0, 0)] * n
        for k in range(n // 2):
            a[k] = a_even[k] + w * a_odd[k]
            a[k + n // 2] = a_even[k] - w * a_odd[k]
            w = w * w_n
        return a

def find_n_th_roots_of_unity(n: int) -> List[complex]:
    """
    Find the n-th roots of unity.
    """
    roots = [cmath.exp(2j * cmath.pi * i / n) for i in range(n)]

    return roots