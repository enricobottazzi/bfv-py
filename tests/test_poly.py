import unittest
import numpy as np
from bfv.polynomial import (
    PolynomialRing,
    Polynomial,
    get_centered_remainder,
)
import random


class TestPolynomialRing(unittest.TestCase):
    def test_init_with_n_and_q(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        quotient = np.array([1, 0, 0, 0, 1])
        self.assertTrue(np.array_equal(Rq.denominator, quotient))
        self.assertEqual(Rq.modulus, q)
        self.assertEqual(Rq.n, n)

    def test_sample_poly_from_rq(self):
        n = 1024
        q = 7
        Rq = PolynomialRing(n, q)
        aq1 = Rq.sample_polynomial()
        aq2 = Rq.sample_polynomial()

        # Ensure that the coefficients of the polynomial are within the range [-(q-1)/2, (q-1)/2]
        lower_bound = -(q - 1) / 2 # inclusive
        upper_bound = (q - 1) / 2 # inclusive
        for coeff in aq1.coefficients:
            self.assertTrue(coeff >= lower_bound and coeff <= upper_bound)
        for coeff in aq2.coefficients:
            self.assertTrue(coeff >= lower_bound and coeff <= upper_bound)

        # Ensure that the degree of the sampled poly is equal or less than d (it might be less if the leading coefficient sampled is 0)
        count1 = 0
        for coeff in aq1.coefficients:
            count1 += 1

        count2 = 0
        for coeff in aq2.coefficients:
            count2 += 1

        self.assertTrue(count1 <= Rq.n)
        self.assertTrue(count2 <= Rq.n)


class TestPolynomialInRingRq(unittest.TestCase):
    def test_init_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients = [7, 1, 0]
        a = Polynomial(coefficients)

        # Reduce the coefficients by the modulus of the polynomial ring
        a.reduce_coefficients_by_modulus(Rq.modulus)

        self.assertTrue(np.array_equal(a.coefficients, [0, 1, 0]))

    def test_add_poly(self):
        coefficients_1 = [3, 3, 4, 4, 4]
        coefficients_2 = [3, 2, 0, 1]
        aq1 = Polynomial(coefficients_1)
        aq2 = Polynomial(coefficients_2)
        result = aq1 + aq2
        assert result.coefficients == [3, 6, 6, 4, 5]


    def test_add_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients_1 = [3, 3, 4, 4, 4]  # r(x)
        r = Polynomial(coefficients_1)

        r.reduce_in_ring(Rq)

        assert r.coefficients == [3, -3, -3, 1]

        coefficients_2 = [3, 3, 2, 0, 1]
        p = Polynomial(coefficients_2)

        # r + p
        result = r + p

        result.reduce_in_ring(Rq)

        assert result.coefficients == [-1, -1, -3, -1]

    def test_mul_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients_1 = [3, 0, 4]
        coefficients_2 = [2, 0, 1]

        aq1 = Polynomial(coefficients_1)
        aq2 = Polynomial(coefficients_2)

        # aq1 * aq2
        result = aq1 * aq2

        result.reduce_in_ring(Rq)

        assert result.coefficients == [0, -3, 0, -2]

    def test_scalar_mul_poly_in_ring_Rq(self):
        n = 4
        q = 7
        Rq = PolynomialRing(n, q)
        coefficients = [4, 3, 0, 4]
        aq1 = Polynomial(coefficients)
        aq2 = Polynomial([2])

        result = aq1 * aq2
        result.reduce_in_ring(Rq)

        assert result.coefficients == [1, -1, 0, 1]

    def test_poly_eval(self):
        # random sample 1024 coefficients in the range 0, 1152921504606584833
        coefficients_1 = []
        for i in range(1024):
            coefficients_1.append(random.randint(0, 1152921504606584833))
        
        coefficients_2 = []
        for i in range(1024):
            coefficients_2.append(random.randint(0, 1152921504606584833))

        coefficients_3 = []
        for i in range(1024):
            coefficients_3.append(random.randint(0, 1152921504606584833))

        aq1 = Polynomial(coefficients_1)
        aq2 = Polynomial(coefficients_2)
        aq3 = Polynomial(coefficients_3)

        # aq1 + aq2 * aq3
        result = aq1 + aq2 * aq3

        # evaluate the polynomial at a random x
        x = random.randint(0, 1152921504606584833)
        result = result.evaluate(x)

        aq1 = aq1.evaluate(x)
        aq2 = aq2.evaluate(x)
        aq3 = aq3.evaluate(x)

        # check if the result is equal to the sum of the two polynomials evaluated at x
        assert result == aq1 + aq2 * aq3

        
class TestCenteredRemainder(unittest.TestCase):
    # For modulus 7, the centered remainder range in which the coefficients of the polynomial should lie is [-3, 3]
    def test_positive_values(self):
        self.assertEqual(
            get_centered_remainder(3, 7), 3
        )
        self.assertEqual(
            get_centered_remainder(15, 7), 1
        )  
        self.assertEqual(
            get_centered_remainder(6, 7), -1
        )  # 6 % 7 = 6, which is > 3. So, 6 - 7 = -1

    def test_negative_values(self):
        self.assertEqual(get_centered_remainder(-8, 7), -1)
        self.assertEqual(
            get_centered_remainder(-15, 7), -1
        ) 
        self.assertEqual(
            get_centered_remainder(-17, 7), -3
        )

    def test_boundary_values(self):
        q = 7
        self.assertEqual(
            get_centered_remainder(- (q - 1) / 2, q), - (q - 1) / 2, q
        )
        self.assertEqual(
            get_centered_remainder((q - 1) / 2, q), (q - 1) / 2
        )

    def test_zero(self):
        self.assertEqual(
            get_centered_remainder(0, 7), 0
        )
