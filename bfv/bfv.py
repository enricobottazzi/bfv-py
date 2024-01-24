from bfv.crt import CRTModuli
from .polynomial import PolynomialRing, Polynomial
from .discrete_gauss import DiscreteGaussian
from .utils import mod_inverse
import numpy as np


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

        # Ensure that t is a prime number
        assert self.is_prime(t), "modulus t must be a prime number"

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

    def is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, n):
            if n % i == 0:
                return False
        return True


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
        self, s: Polynomial, e: Polynomial
    ) -> tuple[Polynomial, Polynomial]:
        """
        Generate a public key from a given secret key.

        Parameters:
        - s: Secret key.
        - e: error polynomial sampled from the distribution χ Error.

        Returns: Generated public key.
        """
        # Sample a polynomial a from Rq
        a = self.rlwe.Rq.sample_polynomial()
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
        error: tuple[Polynomial, Polynomial],
        u: Polynomial,
        q: int,
    ) -> tuple[Polynomial, Polynomial]:
        """
        Encrypt a given message m with a given public_key .

        Parameters:
        - public_key: Public key. The public key must be a tuple of polynomials living in the ring of self.rlwe.Rq.
        - m: message. This must be a polynomial in Rt.
        - error: tuple of error values used in encryption. These must be polynomial sampled from the distribution χ Error.
        - u: ephermeral key polynomial sampled from the distribution χ Ternary.
        - q: modulus q of the ciphertext space. When using the Chinese Remainder Theorem, this is the modulus of the largest ciphertext space.

        Returns:
        ciphertext: Generated ciphertext.
        """
        # Polynomials e0, e1 are sampled the distribution χ Error
        e0 = error[0]
        e1 = error[1]

        # Compute the ciphertext.
        # Q[m]
        q_m = Polynomial([q]) * m

        # Q[m]/t rounded to the nearest integer
        scaled_message = Polynomial(
            [round(coeff / self.rlwe.Rt.modulus) for coeff in q_m.coefficients]
        )

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
        a: Polynomial,
        m: Polynomial,
        e: Polynomial,
        q: int,
    ) -> tuple[Polynomial, Polynomial]:
        """
        Encrypt a given message m with a given secret key .

        Parameters:
        - secret_key: Polynomial sampled from the ternary distribution χ Ternary.
        - a: polynomial sampled from the ring Rq. When using the Chinese Remainder Theorem, this is the polynomial sampled from the small ciphertext space.
        - m: message. This must be a polynomial in Rt.
        - e: error polynomial sampled from the distribution χ Error.
        - q: modulus q of the ciphertext space. When using the Chinese Remainder Theorem, this is the modulus of the largest ciphertext space.

        Returns:
        ciphertext: Generated ciphertext.
        """

        # k^{0} = -t^{-1} namely the multiplicative inverse of t modulo q
        k0 = mod_inverse(self.rlwe.Rt.modulus, self.rlwe.Rq.modulus) * (-1)

        # k^{1} = [QM]t namely the scaled message polynomial
        k1 = Polynomial([q]) * m

        # reduce k^{1} in Rt
        k1.reduce_in_ring(self.rlwe.Rt)

        # a * s
        mul = a * secret_key

        # b = a*s + e.
        b = mul + e

        # ct_0 = b + k^{0}k^{1}
        ct_0 = b + (Polynomial([k0]) * k1)

        # ct_0 will be in Rq
        ct_0.reduce_in_ring(self.rlwe.Rq)

        # ct_1 = -a
        ct_1 = a * Polynomial([-1])

        ciphertext = (ct_0, ct_1)

        return ciphertext

    def PubKeyEncryptConst(
        self,
        public_key: tuple[Polynomial, Polynomial],
        m: Polynomial,
        u: Polynomial,
        delta: int,
    ):
        """
        Encrypt a given message m with a given public_key setting e0 and e1 to 0. This is used for the constant multiplication and addition.

        Parameters:
        - public_key: Public key.
        - m: message.
        - u: ephermeral key polynomial sampled from the distribution χ Ternary.
        - delta: delta = q/t

        Returns:
        ciphertext: Generated ciphertext.
        """

        # Compute the ciphertext.
        # delta * m
        delta_m = Polynomial([delta]) * m

        # pk0 * u
        pk0_u = public_key[0] * u

        # ct_0 = delta * m + pk0 * u
        ct_0 = delta_m + pk0_u

        # ct_0 will be in Rq
        ct_0.reduce_in_ring(self.rlwe.Rq)

        # ct_1 = pk1 * u
        ct_1 = public_key[1] * u

        # ct_1 will be in Rq
        ct_1.reduce_in_ring(self.rlwe.Rq)

        ciphertext = (ct_0, ct_1)

        return ciphertext

    def PubKeyDecrypt(
        self,
        secret_key: Polynomial,
        ciphertext: tuple[Polynomial, Polynomial],
        error: tuple[Polynomial, Polynomial],
        e: Polynomial,
        u: Polynomial,
    ):
        """
        Decrypt a given ciphertext (encrypted using public key encryption) with a given secret key.

        Parameters:
        - secret_key: Secret key.
        - ciphertext: Ciphertext.
        - error: tuple of error values used in encryption. This is used when calculating that the noise is small enough to decrypt the message.
        - e: error polynomial sampled from the distribution χ Error. Used for public key generation. This is used when calculating that the noise is small enough to decrypt the message.
        - u: ephermeral key polynomial sampled from the distribution χ Ternary. Used for encryption. This is used when calculating that the noise is small enough to decrypt the message.

        Returns: Decrypted message.
        """
        # dec = round(t/q * ((ct0 + ct1*s))
        ct0 = ciphertext[0]
        ct1 = ciphertext[1]
        s = secret_key
        t = self.rlwe.Rt.modulus
        q = self.rlwe.Rq.modulus

        # Ensure that all the errors v < q/(2t) - 1/2 (check section 3.1 of 2021/204)
        # v = u * e + e0 + s * e1
        u_e = u * e
        s_e1 = s * error[1]

        v = u_e + error[0]
        v = v + s_e1

        threshold = q / (2 * t) - 1 / 2

        for coeff in v.coefficients:
            assert abs(coeff) < (
                threshold
            ), f"Noise {abs(coeff)} exceeds the threshold value {threshold}, decryption won't work"

        ct1_s = ct1 * s

        # ct0 + ct1*s
        numerator_1 = ct0 + ct1_s

        # Reduce numerator_1 in Rq
        numerator_1.reduce_in_ring(self.rlwe.Rq)

        numerator = Polynomial([t]) * numerator_1

        # For each coefficient of the numerator, divide it by q and round it to the nearest integer
        quotient = [round(coeff / q) for coeff in numerator.coefficients]

        # trim leading zeros
        quotient = np.trim_zeros(quotient, "f")

        quotient_poly = Polynomial(quotient)

        # Reduce the quotient in Rt
        quotient_poly.reduce_in_ring(self.rlwe.Rt)

        return quotient_poly

    def SecretKeyDecrypt(
        self,
        secret_key: Polynomial,
        ciphertext: tuple[Polynomial, Polynomial],
        e: Polynomial,
    ):
        """
        Decrypt a given ciphertext (encrypted using secret key encryption) with a given secret key.

        Parameters:
        - secret_key: Secret key.
        - ciphertext: Ciphertext.
        - e: error polynomial sampled from the distribution χ Error. Used for encryption. This is used when calculating that the noise is small enough to decrypt the message.

        Returns: Decrypted message.
        """
        # dec = round(t/q * ((ct0 + ct1*s))
        ct0 = ciphertext[0]
        ct1 = ciphertext[1]
        s = secret_key
        t = self.rlwe.Rt.modulus
        q = self.rlwe.Rq.modulus

        # Ensure that `e` < Q/(2t)
        threshold = q / (2 * t)

        for coeff in e.coefficients:
            assert abs(coeff) < (
                threshold
            ), f"Noise {abs(coeff)} exceeds the threshold value {threshold}, decryption won't work"

        ct1_s = ct1 * s

        # ct0 + ct1*s
        numerator_1 = ct0 + ct1_s

        # Reduce numerator_1 in Rq
        numerator_1.reduce_in_ring(self.rlwe.Rq)

        numerator = Polynomial([t]) * numerator_1

        # For each coefficient of the numerator, divide it by q and round it to the nearest integer
        quotient = [round(coeff / q) for coeff in numerator.coefficients]

        # trim leading zeros
        quotient = np.trim_zeros(quotient, "f")

        quotient_poly = Polynomial(quotient)

        # Reduce the quotient in Rt
        quotient_poly.reduce_in_ring(self.rlwe.Rt)

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

class BFV_CRT:
    def __init__(self, rlwe_q: RLWE, rlwe_qis: list[RLWE]):
        """
        Initialize a BFV instance where the ciphertext space is represented using the Chinese Remainder Theorem (CRT).

        Parameters:
        - rlwe_q: RLWE instance for the ciphertext space q.
        - rlwe_qis: List of RLWE instances for the ciphertext spaces qi.
        """
        self.bfv_q = BFV(rlwe_q)
        self.bfv_qis = [BFV(rlwe_qi) for rlwe_qi in rlwe_qis]
