from .polynomial import PolynomialRing, Polynomial
from .discrete_gauss import DiscreteGaussian
import numpy as np
import math


class RLWE:
    def __init__(self, n: int, q: int, t: int, distribution: DiscreteGaussian):
        """
        Initialize a RLWE instance starting from the parameters

        Parameters:
        - n: degree of the f(x) which is the denominator of the polynomial ring, must be a power of 2.
        - q: modulus q of the ciphertext space
        - t: modulus t of the plaintext space
        - distribution: Error distribution (e.g. Gaussian).
        """
        # Ensure that the modulus of the plaintext space is smaller than the modulus of the polynomial ring
        if t > q:
            raise ValueError(
                "The modulus of the plaintext space must be smaller than the modulus of the polynomial ring."
            )

        # Ensure that n is a power of 2
        assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"

        # Ensure that p and q are greater than 1
        assert q > 1, "modulus q must be > 1"
        assert t > 1, "modulus t must be > 1"

        # Ensure that q and t are coprime
        assert np.gcd(q, t) == 1, "modulus q and t must be coprime"

        self.n = n
        self.Rq = PolynomialRing(n, q)
        self.Rt = PolynomialRing(n, t)
        self.distribution = distribution

    def SampleFromTernaryDistribution(self) -> Polynomial:
        """
        Sample a polynomial from the χ Ternary distribution.
        Namely, the coefficients are sampled uniformely from the ternary set {-1, 0, 1}. (coefficients are either of them)

        Returns: Sampled polynomial.
        """

        coefficients = np.random.choice([-1, 0, 1], size=self.n)

        coefficients_int = [int(coeff) for coeff in coefficients]

        return Polynomial(coefficients_int)

    def SampleFromErrorDistribution(self) -> Polynomial:
        """
        Sample a polynomial from the χ Error distribution.

        Returns: Sampled polynomial.
        """
        # Sample a polynomial from the Error distribution
        coefficients = self.distribution.sample(self.n)

        coefficients_int = [int(coeff) for coeff in coefficients]

        return Polynomial(coefficients_int)

class BFV:
    def __init__(self, rlwe: RLWE):
        """
        Initialize a BFV instance starting from a rlwe instance

        Parameters:
        - rlwe: RLWE instance.
        """
        self.rlwe = rlwe

    def SecretKeyGen(self) -> Polynomial:
        """
        Randomly generate a secret key.

        Returns: Generated secret key polynomial.
        """

        return self.rlwe.SampleFromTernaryDistribution()

    def PublicKeyGen(
        self, s: Polynomial
    ) -> tuple[Polynomial, Polynomial]:
        """
        Generate a public key from a given secret key.

        Parameters:
        - s: Secret key.

        Returns: Generated public key.
        """        
        # Sample a polynomial a from Rq
        a = self.rlwe.Rq.sample_polynomial()

        # Sample a polynomial e from the distribution χ Error
        e = self.rlwe.SampleFromErrorDistribution()

        # a * s
        mul = a * s

        # b = a*s + e.
        b = mul + e

        # pk0 is a polynomial in Rq
        pk0 = b
        pk0.reduce_in_ring(self.rlwe.Rq)

        # pk1 = -a.
        pk1 = a * Polynomial([-1])

        public_key = (pk0, pk1)

        return public_key

    def PubKeyEncrypt(
        self,
        public_key: tuple[Polynomial, Polynomial],
        m: Polynomial,
    ) -> tuple[Polynomial, Polynomial]:
        """
        Encrypt a given message m with a given public_key .

        Parameters:
        - public_key: Public key. The public key must be a tuple of polynomials living in the ring of self.rlwe.Rq.
        - m: message. This must be a polynomial in Rt.

        Returns:
        ciphertext: Generated ciphertext.
        """
        # Polynomials e0, e1 are sampled the distribution χ Error
        e0 = self.rlwe.SampleFromErrorDistribution()
        e1 = self.rlwe.SampleFromErrorDistribution()

        # Polynomials u is sampled from the distribution χ Ternary
        u = self.rlwe.SampleFromTernaryDistribution()

        # Scale the plaintext message up by delta
        # obtain delta by rounding down q/t to the nearest integer
        delta = math.floor(self.rlwe.Rq.modulus / self.rlwe.Rt.modulus)

        # scaled_message = delta * m
        scaled_message = Polynomial([delta]) * m

        # pk0 * u
        pk0_u = public_key[0] * u

        # scaled_message + pk0 * u + e0
        ct_0 = scaled_message + pk0_u + e0

        # ct_0 will be in Rq
        ct_0.reduce_in_ring(self.rlwe.Rq)

        # pk1 * u
        pk1_u = public_key[1] * u

        # pk1 * u + e1
        ct_1 = pk1_u + e1

        # The result will be in Rq
        ct_1.reduce_in_ring(self.rlwe.Rq)

        ciphertext = (ct_0, ct_1)

        return ciphertext

    def SecretKeyEncrypt(
        self,
        secret_key: Polynomial,
        m: Polynomial,
    ) -> tuple[Polynomial, Polynomial]:
        """
        Encrypt a given message m with a given secret key .

        Parameters:
        - secret_key: Polynomial sampled from the ternary distribution χ Ternary.
        - a: polynomial sampled from the ring Rq. When using the Chinese Remainder Theorem, this is the polynomial sampled from the small ciphertext space.
        - m: message. This must be a polynomial in Rt.

        Returns:
        ciphertext: Generated ciphertext.
        """

        a = self.rlwe.Rq.sample_polynomial()
        e = self.rlwe.SampleFromErrorDistribution()

        delta = math.floor(self.rlwe.Rq.modulus / self.rlwe.Rt.modulus)

        # scaled_message = delta * m
        scaled_message = Polynomial([delta]) * m

        # a * s
        mul = a * secret_key

        # b = a*s + e.
        b = mul + e

        # ct_0 = a*s + e + delta * m
        ct_0 = b + scaled_message

        # ct_0 will be in Rq
        ct_0.reduce_in_ring(self.rlwe.Rq)

        # ct_1 = -a
        ct_1 = a * Polynomial([-1])

        ciphertext = (ct_0, ct_1)

        return (ciphertext)

    def PubKeyEncryptConst(
        self,
        public_key: tuple[Polynomial, Polynomial],
        m: Polynomial,
    ):
        """
        Encrypt a given message m with a given public_key setting e0 and e1 to 0. This is used for the constant multiplication and addition.

        Parameters:
        - public_key: Public key.
        - m: message.

        Returns:
        ciphertext: Generated ciphertext.
        """
        # Polynomials u is sampled from the distribution χ Ternary
        u = self.rlwe.SampleFromTernaryDistribution()

        # Scale the plaintext message up by delta
        # obtain delta by rounding down q/t to the nearest integer
        delta = math.floor(self.rlwe.Rq.modulus / self.rlwe.Rt.modulus)

        # scaled_message = delta * m
        scaled_message = Polynomial([delta]) * m

        # pk0 * u
        pk0_u = public_key[0] * u

        # scaled_message + pk0 * u 
        ct_0 = scaled_message + pk0_u 

        # ct_0 will be in Rq
        ct_0.reduce_in_ring(self.rlwe.Rq)

        # pk1 * u
        pk1_u = public_key[1] * u

        # pk1 * u 
        ct_1 = pk1_u 

        # The result will be in Rq
        ct_1.reduce_in_ring(self.rlwe.Rq)

        ciphertext = (ct_0, ct_1)

        return ciphertext

    def PubKeyDecrypt(
        self,
        secret_key: Polynomial,
        ciphertext: tuple[Polynomial, Polynomial],
    ):
        """
        Decrypt a given ciphertext (encrypted using public key encryption) with a given secret key.

        Parameters:
        - secret_key: Secret key.
        - ciphertext: Ciphertext.

        Returns: Decrypted message.
        """

        ct0 = ciphertext[0]
        ct1 = ciphertext[1]
        s = secret_key
        t = self.rlwe.Rt.modulus
        q = self.rlwe.Rq.modulus

        ct1_s = ct1 * s

        # ct0 + ct1*s
        numerator = ct0 + ct1_s

        # reduce the numerator in Rq
        numerator.reduce_in_ring(self.rlwe.Rq)

        # delta = q/t
        delta = math.floor(q / t)

        # scale the message down by delta and remove the noise performing the rounding by the nearest integer
        quotient = [round(coeff / delta) for coeff in numerator.coefficients]

        # trim leading zeros
        quotient = np.trim_zeros(quotient, "f")

        quotient_poly = Polynomial(quotient)

        return quotient_poly

    def SecretKeyDecrypt(
        self,
        secret_key: Polynomial,
        ciphertext: tuple[Polynomial, Polynomial],
    ):
        """
        Decrypt a given ciphertext (encrypted using secret key encryption) with a given secret key.

        Parameters:
        - secret_key: Secret key.
        - ciphertext: Ciphertext.

        Returns: Decrypted message.
        """
        ct0 = ciphertext[0]
        ct1 = ciphertext[1]
        s = secret_key
        t = self.rlwe.Rt.modulus
        q = self.rlwe.Rq.modulus

        ct1_s = ct1 * s

        # ct0 + ct1*s
        numerator = ct0 + ct1_s

        # reduce the numerator in Rq
        numerator.reduce_in_ring(self.rlwe.Rq)

        # delta = q/t
        delta = math.floor(q / t)

        # scale the message down by delta and remove the noise performing the rounding by the nearest integer
        quotient = [round(coeff / delta) for coeff in numerator.coefficients]

        # trim leading zeros
        quotient = np.trim_zeros(quotient, "f")

        quotient_poly = Polynomial(quotient)

        return quotient_poly

    def EvalAdd(
        self,
        ciphertext1: tuple[Polynomial, Polynomial],
        ciphertext2: tuple[Polynomial, Polynomial],
    ):
        """
        Add two ciphertexts.

        Parameters:
        - ciphertext1: First ciphertext.
        - ciphertext2: Second ciphertext.

        Returns:
        ciphertext_sum: Sum of the two ciphertexts.
        """
        # ct1_0 + ct2_0
        ct0 = ciphertext1[0] + ciphertext2[0]
        ct0.reduce_in_ring(self.rlwe.Rq)

        # ct1_1 + ct2_1
        ct1 = ciphertext1[1] + ciphertext2[1]
        ct1.reduce_in_ring(self.rlwe.Rq)

        return (ct0, ct1)