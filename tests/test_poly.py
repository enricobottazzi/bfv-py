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
        n = 4
        q = 8
        Rq = PolynomialRing(n, q)
        aq1 = Rq.sample_polynomial()
        aq2 = Rq.sample_polynomial()

        # Ensure that the coefficients of the polynomial are within Z_q = (-q/2, q/2]
        for coeff in aq1.coefficients:
            self.assertTrue(coeff > -q // 2 and coeff <= q // 2)
        for coeff in aq2.coefficients:
            self.assertTrue(coeff > -q // 2 and coeff <= q // 2)

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
        coefficients_2 = [3, 3, 2, 0, 1]
        aq1 = Polynomial(coefficients_1)
        aq2 = Polynomial(coefficients_2)
        result = aq1 + aq2
        assert result.coefficients == [6, 6, 6, 4, 5]

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
        # random sample 100 coefficients in the range 0, 1152921504606584833
        coefficients_1 = []
        for i in range(1024):
            coefficients_1.append(random.randint(0, 1152921504606584833))
        
        coefficients_2 = []
        for i in range(1024):
            coefficients_2.append(random.randint(0, 1152921504606584833))

        aq1 = Polynomial(coefficients_1)
        aq2 = Polynomial(coefficients_2)

        # aq1 * aq2
        result = aq1 * aq2

        # evaluate the polynomial at a random x
        x = random.randint(0, 1152921504606584833)
        result = result.evaluate(x)

        aq1 = aq1.evaluate(x)
        aq2 = aq2.evaluate(x)

        # check if the result is equal to the product of the two polynomials evaluated at x
        assert result == aq1 * aq2


class TestCenteredRemainder(unittest.TestCase):
    def test_positive_values(self):
        self.assertEqual(
            get_centered_remainder(7, 10), -3
        )  # 7 % 10. Lies in the range (-5, 5]
        self.assertEqual(
            get_centered_remainder(15, 10), 5
        )  # 15 % 10 = 5, which is <= 5
        self.assertEqual(
            get_centered_remainder(17, 10), -3
        )  # 17 % 10 = 7, which is > 5. So, 7 - 10 = -3

    def test_negative_values(self):
        self.assertEqual(get_centered_remainder(-7, 10), 3)  # Lies in the range (-5, 5]
        self.assertEqual(
            get_centered_remainder(-15, 10), 5
        )  # -15 % 10 = 5 (in Python, % returns non-negative), which is <= 5
        self.assertEqual(
            get_centered_remainder(-17, 10), 3
        )  # -17 % 10 = 3, which is <= 5

    def test_boundary_values(self):
        q = 7
        self.assertEqual(
            get_centered_remainder(-q // 2 + 1, q), -q // 2 + 1
        )  # The smallest positive number in the range
        self.assertEqual(
            get_centered_remainder(q // 2, q), q // 2
        )  # The largest number in the range

    def test_zero(self):
        self.assertEqual(
            get_centered_remainder(0, 10), 0
        )  # 0 lies in the range (-5, 5]
