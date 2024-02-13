from typing import List
import galois
from bfv.polynomial import get_centered_remainder, Polynomial
from bfv.utils import adjust_negative_coefficients

def ntt_poly_mul(coeffs1: List[int], coeffs2: List[int], size: int, p: int) -> List[int]:
    """
    Multiply two polynomials using the NTT with prime p
    size is the size of the NTT tranform such that p = size * c + 1 where c is a constant
    """

    ntt_evals_1 = galois.ntt(coeffs1, size, p)
    ntt_evals_2 = galois.ntt(coeffs2, size, p)
    ntt_product = [ntt_evals_1[i] * ntt_evals_2[i] for i in range(size)]
    product = galois.intt(ntt_product, size, p)
    
    # trim the product to the correct length
    product = product[:len(coeffs1) + len(coeffs2) - 1]

    return product

def ntt_poly_mul_centered_remainder(coeffs_1: List[int], coeffs_2: List[int], size: int, p: int) -> List[int]:
    """
    Multiply two polynomials using the NTT with prime p
    size is the size of the NTT tranform such that p = size * c + 1 where c is a constant
    Polynomial coefficients are defined with their centered remainder modulo p
    """

    coeffs_1_normalized = adjust_negative_coefficients(Polynomial(coeffs_1), p)
    coeffs_2_normalized = adjust_negative_coefficients(Polynomial(coeffs_2), p)

    product_normalized = ntt_poly_mul(coeffs_1_normalized.coefficients, coeffs_2_normalized.coefficients, size, p)

    product_centered = [get_centered_remainder(int(coeff), p) for coeff in product_normalized]

    return product_centered