import unittest
from bfv.fft import find_n_th_roots_of_unity, recursive_fft, recursive_ifft
from bfv.polynomial import poly_mul_naive
import random
import copy
from mpmath import *
mp.dps = 50

class TestFFT(unittest.TestCase):
    def test_roots_of_unity(self):
        n = 3
        roots = find_n_th_roots_of_unity(n)

        self.assertAlmostEqual(roots[0], complex(1, 0))
        self.assertAlmostEqual(roots[1], complex(-0.5, sqrt(3) / 2))
        self.assertAlmostEqual(roots[2], complex(-0.5, -sqrt(3) / 2))

    def test_ifft(self):
        coeffs = [random.getrandbits(60) for _ in range(1024)]

        # turn the polynomial into its point form using FFT O(n log n)
        fft_evals = recursive_fft(coeffs)

        # turn the polynomial back into its coefficient form using IFFT O(n log n)
        ifft_coeffs = recursive_ifft(fft_evals)

        assert ifft_coeffs == coeffs   

    def test_poly_mul_fft(self):
        deg1 = random.randint(0, 1000) 
        deg2 = random.randint(0, 1000)
    
        # generate random coefficients for two polynomials
        coeffs1 = [random.getrandbits(60) for _ in range(deg1 + 1)]
        coeffs2 = [random.getrandbits(60) for _ in range(deg2 + 1)]

        product_len = len(coeffs1) + len(coeffs2) - 1

        # pad the coefficients with zeroes at the beginning to make them the same length of product_len (https://math.stackexchange.com/questions/764727/concrete-fft-polynomial-multiplication-example/764870#764870)
        # that's because we need to be able to compute #product_len points during convolution
        coeffs1_padded = [0] * (product_len - len(coeffs1)) + coeffs1
        coeffs2_padded = [0] * (product_len - len(coeffs2)) + coeffs2

        # fft works when the length of the coefficients is a power of 2
        n = 1
        while n < product_len:
            n *= 2
        
        # further pad the coefficients with zeroes at the beginning to make them of length n (power of two)
        coeffs1_padded = [0] * (n - product_len) + coeffs1_padded
        coeffs2_padded = [0] * (n - product_len) + coeffs2_padded

        coeffs1_reversed = copy.deepcopy(coeffs1_padded)
        coeffs2_reversed = copy.deepcopy(coeffs2_padded)

        coeffs1_reversed.reverse()
        coeffs2_reversed.reverse()

        # turn the polynomials into their point form using FFT O(n log n)
        fft_evals1 = recursive_fft(coeffs1_reversed)
        fft_evals2 = recursive_fft(coeffs2_reversed)

        # multiply the polynomials in point form to get the product in point form O(n)
        fft_product_evals = [fft_evals1[i] * fft_evals2[i] for i in range(n)]

        # turn the product back into its coefficient form using IFFT O(n log n)
        product_coeffs = recursive_ifft(fft_product_evals)

        # calculate the padding for product_coeffs
        product_padding = len(product_coeffs) - product_len
        
        # remove the last padding zeroes
        product_coeffs_no_pad = product_coeffs[:-product_padding]

        # reverse the product_coeffs_no_pad list to obtain an array in which the first element is the highest degree coefficient
        product_coeffs_no_pad.reverse()

        # perform the multiplication between `coeffs1` and `coeffs2` naively
        product_naive = poly_mul_naive(coeffs1, coeffs2)

        for i in range(len(product_coeffs_no_pad)):
            assert product_coeffs_no_pad[i] == product_naive[i]




