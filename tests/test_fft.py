import unittest
from bfv.fft import find_n_th_roots_of_unity, recursive_fft, recursive_ifft
import math
import random

class TestFFT(unittest.TestCase):
    def test_roots_of_unity(self):
        n = 3
        roots = find_n_th_roots_of_unity(n)

        self.assertAlmostEqual(roots[0], complex(1, 0))
        self.assertAlmostEqual(roots[1], complex(-0.5, math.sqrt(3) / 2))
        self.assertAlmostEqual(roots[2], complex(-0.5, -math.sqrt(3) / 2))

    def test_fft(self):    
        # generate random coefficients
        coeffs = [random.randint(0, 8000) for _ in range(2048)]
        n = len(coeffs)
        roots = find_n_th_roots_of_unity(n)

        # turn the polynomial into its point form naively O(n^2)
        evals = []
        for root in roots:
            eval = 0
            for i in range(n):
                eval += coeffs[i] * root ** i
            evals.append(eval)

        # turn the polynomial into its point form using FFT O(n log n)
        fft_evals = recursive_fft(coeffs)

        # round the real part of the complex numbers
        evals = [round(eval.real) for eval in evals]
        fft_evals = [round(fft_eval.real) for fft_eval in fft_evals]

        # assert that the two methods give the same result
        assert evals == fft_evals

    def test_ifft(self):
        # generate random coefficients
        coeffs = [random.randint(0, 8000) for _ in range(1024)]

        # turn the polynomial into its point form using FFT O(n log n)
        fft_evals = recursive_fft(coeffs)

        # turn the polynomial back into its coefficient form using IFFT O(n log n)
        ifft_coeffs = recursive_ifft(fft_evals)
                
        # divide the coefficients by n
        ifft_coeffs = [ifft_coeff / len(coeffs) for ifft_coeff in ifft_coeffs]       

        # round the real part of the complex numbers
        recovered_coeffs = [round(ifft_coeff.real) for ifft_coeff in ifft_coeffs] 

        # assert that the two methods give the same result
        for i in range(len(coeffs)):
            assert coeffs[i] == recovered_coeffs[i]


