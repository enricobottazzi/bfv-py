import unittest
from bfv.fft import find_n_th_roots_of_unity, recursive_fft, recursive_ifft
from bfv.polynomial import poly_mul
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

    def test_poly_mul_fft(self):
        # generate random coefficients
        coeffs1 = [random.randint(0, 8000) for _ in range(1024)]
        coeffs2 = [random.randint(0, 8000) for _ in range(1024)]

        product_len = len(coeffs1) + len(coeffs2) - 1

        # pad the coefficients with zeroes at the beginning to make them the same length of product_len
        coeffs1 = [0] * (product_len - len(coeffs1)) + coeffs1
        coeffs2 = [0] * (product_len - len(coeffs2)) + coeffs2

        # fft works when the length of the coefficients is a power of 2
        n = 1
        while n < product_len:
            n *= 2
        
        # further pad the coefficients with zeroes at the beginning to make them the same length of n
        coeffs1 = [0] * (n - product_len) + coeffs1
        coeffs2 = [0] * (n - product_len) + coeffs2

        assert len(coeffs1) == n
        assert len(coeffs2) == n

        # turn the polynomials into their point form using FFT O(n log n)
        fft_evals1 = recursive_fft(coeffs1)
        fft_evals2 = recursive_fft(coeffs2)

        # multiply the polynomials in point form to get the product in point form O(n)
        fft_product_evals = [fft_evals1[i] * fft_evals2[i] for i in range(n)]

        # turn the product back into its coefficient form using IFFT O(n log n)
        product_coeffs = recursive_ifft(fft_product_evals)

        # divide the coefficients by n
        product_coeffs = [coeff / n for coeff in product_coeffs]       

        # round the real part of the complex numbers
        product_coeffs = [round(coeff.real) for coeff in product_coeffs]
        
        # perform naive polynomial multiplication
        product_naive = poly_mul(coeffs1, coeffs2)

        # remove the leading zeroes from product_naive
        while product_naive[0] == 0:
            product_naive = product_naive[1:]

        # remove the last zeroes from product_coeffs
        while product_coeffs[-1] == 0:
            product_coeffs = product_coeffs[:-1]

        # assert that the two methods give the same result
        assert product_naive == product_coeffs