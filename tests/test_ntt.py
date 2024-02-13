import unittest
import random
import galois
from bfv.ntt import ntt_poly_mul, ntt_poly_mul_centered_remainder
from bfv.polynomial import Polynomial, get_centered_remainder, poly_mul_naive
from bfv.utils import adjust_negative_coefficients

class TestNTT(unittest.TestCase):

    def test_ntt(self):
        p = 1152921504606584833
        k = 4
        coeffs = [random.randint(0, p - 1) for _ in range(2**k)]
        ntt_evals = galois.ntt(coeffs, 2**k, p)
        ntt_coeffs = galois.intt(ntt_evals, 2**k, p)
        for i in range(2**k):
            assert ntt_coeffs[i] == coeffs[i]


    def test_poly_mul_ntt(self):
        p = 1152921504606584833
        k = 4
        coeffs_1 = [random.randint(0, p - 1) for _ in range(2**(k-1))]
        coeffs_2 = [random.randint(0, p - 1) for _ in range(2**(k-1))]

        # multiply the polynomials naively
        product_naive = poly_mul_naive(coeffs_1, coeffs_2)

        # reduce the coefficients modulo p
        for i in range(len(product_naive)):
            product_naive[i] = product_naive[i] % p

        ntt_evals_1 = galois.ntt(coeffs_1, 2**k, p)
        ntt_evals_2 = galois.ntt(coeffs_2, 2**k, p)

        # perform point wise multiplication
        ntt_product = [ntt_evals_1[i] * ntt_evals_2[i] for i in range(2**k)]

        # perform the inverse NTT
        product = galois.intt(ntt_product, 2**k, p)

        # trim the product to the correct length
        product = product[:len(coeffs_1) + len(coeffs_2) - 1]

        for i in range(len(product)):
            assert product[i] == product_naive[i]

    def test_ntt_poly_mul_compact(self):
        p = 1152921504606584833
        k = 4
        coeffs_1 = [random.randint(0, p - 1) for _ in range(2**(k-1))]
        coeffs_2 = [random.randint(0, p - 1) for _ in range(2**(k-1))]
    
        product = ntt_poly_mul(coeffs_1, coeffs_2, 2**k, p)

        # multiply the polynomials naively
        product_naive = poly_mul_naive(coeffs_1, coeffs_2)

        # reduce the coefficients modulo p
        for i in range(len(product_naive)):
            product_naive[i] = product_naive[i] % p

        for i in range(len(product)):
            assert product[i] == product_naive[i]

    def test_ntt_poly_mul_centered_remainder(self):
        p = 1152921504606584833
        k = 4
        coeffs_1 = [random.randint(0, p - 1) for _ in range(2**(k-1))]
        coeffs_2 = [random.randint(0, p - 1) for _ in range(2**(k-1))]

        coeffs_1_centered = [get_centered_remainder(coeff, p) for coeff in coeffs_1]
        coeffs_2_centered = [get_centered_remainder(coeff, p) for coeff in coeffs_2]

        product_centered = Polynomial(coeffs_1) * Polynomial(coeffs_2)
        product_centered.reduce_coefficients_by_modulus(p)

        # now perform the multiplication using ntt
        coeffs_1_normalized = adjust_negative_coefficients(Polynomial(coeffs_1_centered), p)
        coeffs_2_normalized = adjust_negative_coefficients(Polynomial(coeffs_2_centered), p)

        product_normalized = ntt_poly_mul(coeffs_1_normalized.coefficients, coeffs_2_normalized.coefficients, 2**k, p)

        # now get the centered remainder of the product_normalized
        product_normalized_centered = [get_centered_remainder(int(coeff), p) for coeff in product_normalized]

        for i in range(len(product_centered.coefficients)):
            assert product_centered.coefficients[i] == product_normalized_centered[i]


    def test_ntt_poly_mul_centered_remainder_compact(self):
        p = 1152921504606584833
        k = 4
        coeffs_1 = [random.randint(0, p - 1) for _ in range(2**(k-1))]
        coeffs_2 = [random.randint(0, p - 1) for _ in range(2**(k-1))]
        product_centered = Polynomial(coeffs_1) * Polynomial(coeffs_2)
        product_centered.reduce_coefficients_by_modulus(p)

        product_centered_ntt = ntt_poly_mul_centered_remainder(coeffs_1, coeffs_2, 2**k, p)

        for i in range(len(product_centered.coefficients)):
            assert product_centered.coefficients[i] == product_centered_ntt[i]
        