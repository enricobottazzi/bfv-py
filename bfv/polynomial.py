import random


class PolynomialRing:
    def __init__(self, n: int, modulus: int) -> None:
        """
        Initialize a polynomial ring R_modulus = Z_modulus[x]/f(x) where f(x)=x^n+1.
        - n is a power of 2.
        """

        assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"

        fx = [1] + [0] * (n - 1) + [1]

        self.denominator = fx
        self.modulus = modulus
        self.n = n

    def sample_polynomial(self) -> "Polynomial":
        """
        Sample polynomial from the ring
        """

        # range for random.randint
        lower_bound = -self.modulus // 2  # exclusive
        upper_bound = self.modulus // 2  # inclusive

        # generate n random coefficients in the range (lower_bound, upper_bound]
        coeffs = [random.randint(lower_bound + 1, upper_bound) for _ in range(self.n)]

        return Polynomial(coeffs)

    def __eq__(self, other) -> bool:
        if isinstance(other, PolynomialRing):
            return (
                self.denominator == other.denominator and self.modulus == other.modulus
            )
        return False


class Polynomial:
    def __init__(self, coefficients: list[int]):
        self.coefficients = coefficients

    def reduce_coefficients_by_modulus(self, modulus: int) -> None:
        """
        Reduce the coefficients of the polynomial by the modulus of the polynomial ring.
        """
        for i in range(len(self.coefficients)):
            self.coefficients[i] = get_centered_remainder(self.coefficients[i], modulus)

    def reduce_coefficients_by_cyclo(self, cyclo: list[int]) -> None:
        """
        Reduce the coefficients by dividing it by the cyclotomic polynomial and returning the remainder.
        The cyclotomic polynomial is x^n+1.
        """
        _, remainder = poly_div(self.coefficients, cyclo)

        n = len(cyclo) - 1

        # pad the remainder with zeroes to make it len=n
        remainder = [0] * (n - len(remainder)) + remainder

        assert len(remainder) == n

        self.coefficients = remainder

    def reduce_in_ring(self, ring: PolynomialRing) -> None:
        """
        Reduce the coefficients of the polynomial by the modulus of the polynomial ring and by the denominator of the polynomial ring.
        """
        self.reduce_coefficients_by_cyclo(ring.denominator)
        self.reduce_coefficients_by_modulus(ring.modulus)

    def __add__(self, other) -> "Polynomial":
        return Polynomial(poly_add(self.coefficients, other.coefficients))

    def __mul__(self, other) -> "Polynomial":
        return Polynomial(poly_mul(self.coefficients, other.coefficients))
    
    def evaluate(self, x: int) -> int:
        """
        Evaluate the polynomial at x.
        """
        result = 0
        for coeff in reversed(self.coefficients):
            result = result * x + coeff
        return result


def poly_div(dividend: list[int], divisor: list[int]) -> tuple[list[int], list[int]]:
    # Initialize quotient and remainder
    quotient = [0] * (len(dividend) - len(divisor) + 1)
    remainder = list(dividend)

    # Main division loop
    for i in range(len(quotient)):
        coeff = (
            remainder[i] // divisor[0]
        )  # Calculate the leading coefficient of quotient
        # turn coeff into an integer
        coeff = coeff
        quotient[i] = coeff

        # Subtract the current divisor*coeff from the remainder
        for j in range(len(divisor)):
            rem = remainder[i + j]
            rem -= divisor[j] * coeff
            remainder[i + j] = rem

    # Remove leading zeroes in remainder, if any
    while remainder and remainder[0] == 0:
        remainder.pop(0)

    return quotient, remainder


def poly_mul(poly1: list[int], poly2: list[int]) -> list[int]:
    # The degree of the product polynomial is the sum of the degrees of the input polynomials
    result_degree = len(poly1) + len(poly2) - 1
    # Initialize the product polynomial with zeros
    product = [0] * result_degree

    # Multiply each term of the first polynomial by each term of the second polynomial
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            product[i + j] += poly1[i] * poly2[j]

    return product


def poly_add(poly1: list[int], poly2: list[int]) -> list[int]:
    # The degree of the sum polynomial is the max of the degrees of the input polynomials
    result_degree = max(len(poly1), len(poly2))
    # Initialize the sum polynomial with zeros
    sum = [0] * result_degree

    for i in range(len(poly1)):
        sum[i + result_degree - len(poly1)] += poly1[i]

    for i in range(len(poly2)):
        sum[i + result_degree - len(poly2)] += poly2[i]

    return sum


def get_centered_remainder(x, modulus) -> int:
    """
    Returns the centered remainder of x with respect to modulus.
    """
    r = x % modulus
    return r if r <= modulus / 2 else r - modulus

