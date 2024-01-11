import math
import unittest
from bfv.crt import CRTModuli, CRTPolynomial
from bfv.discrete_gauss import DiscreteGaussian
from bfv.polynomial import PolynomialRing, Polynomial
from bfv.bfv import RLWE, BFV


class TestRLWE(unittest.TestCase):
    def setUp(self):
        self.n = 1024
        self.q = 536870909
        self.sigma = 3.2
        self.discrete_gaussian = DiscreteGaussian(self.sigma)
        self.t = 257
        self.delta = int(math.floor(self.q / self.t))
        self.rlwe = RLWE(self.n, self.q, self.t, self.discrete_gaussian)

    def test_rlwe_initialization(self):
        self.assertEqual(
            self.rlwe.Rq.denominator, PolynomialRing(self.n, self.q).denominator
        )
        self.assertEqual(self.rlwe.Rq.n, PolynomialRing(self.n, self.q).n)
        self.assertEqual(self.rlwe.Rq.modulus, PolynomialRing(self.n, self.q).modulus)

        self.assertEqual(
            self.rlwe.Rt.denominator, PolynomialRing(self.n, self.t).denominator
        )
        self.assertEqual(self.rlwe.Rt.n, PolynomialRing(self.n, self.t).n)
        self.assertEqual(self.rlwe.Rt.modulus, PolynomialRing(self.n, self.t).modulus)

        self.assertEqual(self.rlwe.distribution, self.discrete_gaussian)

    def test_sample_from_chi_key_distribution(self):
        key = self.rlwe.SampleFromTernaryDistribution()
        # Ensure that the degree of the sampled poly is equal or less than d (it might be less if the leading coefficient sampled is 0)
        self.assertTrue(len(key.coefficients) <= self.rlwe.n)

        # Ensure that the coefficients of the key are within {-1, 0, 1}
        for coeff in key.coefficients:
            self.assertTrue(coeff == -1 or coeff == 0 or coeff == 1)

    def test_sample_from_chi_error_distribution(self):
        error = self.rlwe.SampleFromErrorDistribution()
        # Ensure that the degree of the sample polynomial is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(error.coefficients) <= self.rlwe.n)

        # Ensure that the secret key is sampled from the error distribution by checking that each coefficient is within the range of the error distribution
        for coeff in error.coefficients:
            self.assertTrue(coeff >= -6 * self.sigma and coeff <= 6 * self.sigma)


class TestBFV(unittest.TestCase):
    def setUp(self):
        n = 1024
        q = 536870909
        sigma = 3.2
        discrete_gaussian = DiscreteGaussian(sigma)
        t = 257
        self.delta = int(math.floor(q / t))
        rlwe = RLWE(n, q, t, discrete_gaussian)
        self.bfv = BFV(rlwe)

    def test_secret_key_gen(self):
        secret_key = self.bfv.SecretKeyGen()
        # Ensure that the degree of the sample polynomial is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(secret_key.coefficients) <= self.bfv.rlwe.n)

        # Ensure that the coefficients of the key are within {-1, 0, 1}
        for coeff in secret_key.coefficients:
            self.assertTrue(coeff == -1 or coeff == 0 or coeff == 1)

    def test_public_key_gen(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.rlwe.SampleFromErrorDistribution()
        public_key = self.bfv.PublicKeyGen(secret_key, e)

        self.assertIsInstance(public_key, tuple)
        self.assertEqual(len(public_key), 2)
        self.assertIsInstance(public_key[0], Polynomial)
        self.assertIsInstance(public_key[1], Polynomial)

        # Ensure that the coefficients of the public key are within Z_q = (-q/2, q/2)
        for coeff in public_key[0].coefficients:
            self.assertTrue(
                coeff > -self.bfv.rlwe.Rq.modulus // 2
                and coeff <= self.bfv.rlwe.Rq.modulus // 2
            )
        for coeff in public_key[1].coefficients:
            self.assertTrue(
                coeff > -self.bfv.rlwe.Rq.modulus // 2
                and coeff <= self.bfv.rlwe.Rq.modulus // 2
            )

    def test_message_sample(self):
        message = self.bfv.rlwe.Rt.sample_polynomial()

        # Ensure that the coefficients of the public key are within Z_t = (-t/2, t/2]
        for coeff in message.coefficients:
            self.assertTrue(
                coeff > -self.bfv.rlwe.Rt.modulus // 2
                and coeff <= self.bfv.rlwe.Rt.modulus // 2
            )

    def test_valid_public_key_encryption(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.rlwe.SampleFromErrorDistribution()
        public_key = self.bfv.PublicKeyGen(secret_key, e)

        message = self.bfv.rlwe.Rt.sample_polynomial()

        e0 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv.rlwe.SampleFromErrorDistribution()
        u = self.bfv.rlwe.SampleFromTernaryDistribution()

        ciphertext = self.bfv.PubKeyEncrypt(
            public_key, message, (e0, e1), u, self.bfv.rlwe.Rq.modulus
        )

        # Ensure that the ciphertext is a polynomial in Rq
        # Ensure that the coefficients of the ciphertext are within Z_q = (-q/2, q/2)
        for coeff in ciphertext[0].coefficients:
            self.assertTrue(
                coeff > -self.bfv.rlwe.Rq.modulus // 2
                and coeff <= self.bfv.rlwe.Rq.modulus // 2
            )
        for coeff in ciphertext[1].coefficients:
            self.assertTrue(
                coeff > -self.bfv.rlwe.Rq.modulus // 2
                and coeff <= self.bfv.rlwe.Rq.modulus // 2
            )
        # Ensure that the degree of the ciphertext is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(ciphertext[0].coefficients) <= self.bfv.rlwe.n)

    def test_valid_public_key_decryption(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.rlwe.SampleFromErrorDistribution()
        public_key = self.bfv.PublicKeyGen(secret_key, e)

        message = self.bfv.rlwe.Rt.sample_polynomial()

        e0 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv.rlwe.SampleFromErrorDistribution()
        u = self.bfv.rlwe.SampleFromTernaryDistribution()

        ciphertext = self.bfv.PubKeyEncrypt(
            public_key, message, (e0, e1), u, self.bfv.rlwe.Rq.modulus
        )

        dec = self.bfv.PubKeyDecrypt(secret_key, ciphertext, (e0, e1), e, u)

        # ensure that message and dec are the same
        for i in range(len(message.coefficients)):
            self.assertEqual(message.coefficients[i], dec.coefficients[i])

    def test_valid_secret_key_decryption(self):
        secret_key = self.bfv.SecretKeyGen()
        message = self.bfv.rlwe.Rt.sample_polynomial()
        e = self.bfv.rlwe.SampleFromErrorDistribution()

        ciphertext = self.bfv.SecretKeyEncrypt(
            secret_key, message, e, self.bfv.rlwe.Rq.modulus
        )

        dec = self.bfv.SecretKeyDecrypt(secret_key, ciphertext, e)

        # ensure that message and dec are the same
        for i in range(len(message.coefficients)):
            self.assertEqual(message.coefficients[i], dec.coefficients[i])

    def test_eval_add(self):
        secret_key = self.bfv.SecretKeyGen()

        e = self.bfv.rlwe.SampleFromErrorDistribution()

        public_key = self.bfv.PublicKeyGen(secret_key, e)

        message1 = self.bfv.rlwe.Rt.sample_polynomial()
        message2 = self.bfv.rlwe.Rt.sample_polynomial()
        message_sum = message1 + message2
        message_sum.reduce_in_ring(self.bfv.rlwe.Rt)

        e0 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv.rlwe.SampleFromErrorDistribution()
        u1 = self.bfv.rlwe.SampleFromTernaryDistribution()
        error1 = (e0, e1)
        ciphertext1 = self.bfv.PubKeyEncrypt(
            public_key, message1, error1, u1, self.bfv.rlwe.Rq.modulus
        )

        e0 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv.rlwe.SampleFromErrorDistribution()
        u2 = self.bfv.rlwe.SampleFromTernaryDistribution()
        error2 = (e0, e1)
        ciphertext2 = self.bfv.PubKeyEncrypt(
            public_key, message2, error2, u2, self.bfv.rlwe.Rq.modulus
        )

        ciphertext_sum = self.bfv.EvalAdd(ciphertext1, ciphertext2)

        # Ensure that the ciphertext_sum is a polynomial in Rq
        # Ensure that the ciphertext_sum of the ciphertext are within Z_q = (-q/2, q/2)
        for coeff in ciphertext_sum[0].coefficients:
            self.assertTrue(
                coeff > -self.bfv.rlwe.Rq.modulus // 2
                and coeff <= self.bfv.rlwe.Rq.modulus // 2
            )
        for coeff in ciphertext_sum[1].coefficients:
            self.assertTrue(
                coeff > -self.bfv.rlwe.Rq.modulus // 2
                and coeff <= self.bfv.rlwe.Rq.modulus // 2
            )
        # Ensure that the degree of the ciphertext_sum is at most n-1, which means the has at most n coefficients
        self.assertTrue(len(ciphertext_sum[0].coefficients) <= self.bfv.rlwe.n)

        # decrypt ciphertext_sum
        # Note that the after performing EvalAdd, the error is the sum of the errors of the two ciphertexts
        # As demonstrated in equation 11 of the paper https://inferati.azureedge.net/docs/inferati-fhe-bfv.pdf
        e0_sum = error1[0] + error2[0]
        e1_sum = error1[1] + error2[1]
        e_sum = (e0_sum, e1_sum)

        u_sum = u1 + u2

        dec = self.bfv.PubKeyDecrypt(secret_key, ciphertext_sum, e_sum, e, u_sum)

        # ensure that message_sum and dec are the same
        for i in range(len(message_sum.coefficients)):
            self.assertEqual(message_sum.coefficients[i], dec.coefficients[i])

    def test_eval_const_add(self):
        secret_key = self.bfv.SecretKeyGen()
        e = self.bfv.rlwe.SampleFromErrorDistribution()

        public_key = self.bfv.PublicKeyGen(secret_key, e)

        message1 = self.bfv.rlwe.Rt.sample_polynomial()
        message2 = self.bfv.rlwe.Rt.sample_polynomial()
        message_sum = message1 + message2
        message_sum.reduce_in_ring(self.bfv.rlwe.Rt)

        e0 = self.bfv.rlwe.SampleFromErrorDistribution()
        e1 = self.bfv.rlwe.SampleFromErrorDistribution()
        error = (e0, e1)
        u1 = self.bfv.rlwe.SampleFromTernaryDistribution()

        ciphertext1 = self.bfv.PubKeyEncrypt(
            public_key, message1, error, u1, self.bfv.rlwe.Rq.modulus
        )

        u2 = self.bfv.rlwe.SampleFromTernaryDistribution()

        const_ciphertext = self.bfv.PubKeyEncryptConst(
            public_key, message2, u2, self.delta
        )

        ciphertext_sum = self.bfv.EvalAdd(ciphertext1, const_ciphertext)

        u_sum = u1 + u2

        # decrypt ciphertext_sum
        dec = self.bfv.PubKeyDecrypt(secret_key, ciphertext_sum, error, e, u_sum)

        # ensure that message_sum and dec are the same
        for i in range(len(message_sum.coefficients)):
            self.assertEqual(message_sum.coefficients[i], dec.coefficients[i])


class TestBFVVWithCRT(unittest.TestCase):
    # The bigmodulus q is intepreted as a product of small moduli qis.
    def setUp(self):
        qis = [
            1152921504606584833,
            1152921504598720513,
            1152921504597016577,
            1152921504595968001,
            1152921504595640321,
            1152921504593412097,
            1152921504592822273,
            1152921504592429057,
            1152921504589938689,
            1152921504586530817,
            1152921504585547777,
            1152921504583647233,
            1152921504581877761,
            1152921504581419009,
            1152921504580894721,
        ]
        crt_moduli = CRTModuli(qis)
        self.crt_moduli = crt_moduli
        self.n = 1024
        self.sigma = 3.2
        self.discrete_gaussian = DiscreteGaussian(self.sigma)
        self.t = 65537

    # In the CRT setting, small integers such as noise and key coefficients are drawn from the Error distribution and Ternary distribution respectively and are stored as single precision integers.
    # In this test, the public key is generated in the Rq basis and then transformed to the CRT basis.
    # The encryption operation is performed in the CRT basis (Rqi).
    # Eventually, the ciphertext is recovered in the Rq basis and compared with the ciphertext generated in the Rq basis.
    def test_valid_public_key_encryption(self):
        bfv_rq = BFV(RLWE(self.n, self.crt_moduli.q, self.t, self.discrete_gaussian))
        secret_key = bfv_rq.SecretKeyGen()
        e = bfv_rq.rlwe.SampleFromErrorDistribution()
        u = bfv_rq.rlwe.SampleFromTernaryDistribution()
        e0 = bfv_rq.rlwe.SampleFromErrorDistribution()
        e1 = bfv_rq.rlwe.SampleFromErrorDistribution()
        message = bfv_rq.rlwe.Rt.sample_polynomial()
        public_key = bfv_rq.PublicKeyGen(secret_key, e)

        # Perform encryption in CRT basis
        c0_rqis = []
        c1_rqis = []

        # Transform public key to CRT basis
        pk0_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(
            public_key[0], self.n, self.crt_moduli
        )
        pk1_rqis = CRTPolynomial.from_rq_polynomial_to_rqi_polynomials(
            public_key[1], self.n, self.crt_moduli
        )

        for i in range(len(self.crt_moduli.qis)):
            bfv_rqi = BFV(
                RLWE(self.n, self.crt_moduli.qis[i], self.t, self.discrete_gaussian)
            )
            public_key_rqi = (pk0_rqis[i], pk1_rqis[i])

            ciphertext = bfv_rqi.PubKeyEncrypt(
                public_key_rqi, message, (e0, e1), u, self.crt_moduli.q
            )
            c0_rqis.append(ciphertext[0])
            c1_rqis.append(ciphertext[1])

        # Recover ciphertext in Rq
        c0 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c0_rqis, self.n, self.crt_moduli
        )
        c1 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c1_rqis, self.n, self.crt_moduli
        )

        # Perform encryption in Rq
        ciphertext = bfv_rq.PubKeyEncrypt(
            public_key, message, (e0, e1), u, self.crt_moduli.q
        )

        # Assert that the two ciphertexts are the same
        self.assertEqual(c0.coefficients, ciphertext[0].coefficients)
        self.assertEqual(c1.coefficients, ciphertext[1].coefficients)

    # In the CRT setting, The uniform elements `a` used to generate the public key are chosen directly in the CRT basis by drawing uniform value ai in Rqi.
    # The operations for encryption are implemented directly in the CRT basis.
    # The ciphertext is recovered in the Rq basis and then decrypted in the Rqi basis. The decrypted message is compared with the original message.
    def test_valid_public_key_decryption(self):
        bfv_rq = BFV(RLWE(self.n, self.crt_moduli.q, self.t, self.discrete_gaussian))

        secret_key = bfv_rq.SecretKeyGen()
        e = bfv_rq.rlwe.SampleFromErrorDistribution()
        u = bfv_rq.rlwe.SampleFromTernaryDistribution()
        e0 = bfv_rq.rlwe.SampleFromErrorDistribution()
        e1 = bfv_rq.rlwe.SampleFromErrorDistribution()
        message = bfv_rq.rlwe.Rt.sample_polynomial()

        # Perform encryption in CRT basis
        c0_rqis = []
        c1_rqis = []

        for i in range(len(self.crt_moduli.qis)):
            bfv_rqi = BFV(
                RLWE(self.n, self.crt_moduli.qis[i], self.t, self.discrete_gaussian)
            )
            public_key_rqi = bfv_rqi.PublicKeyGen(secret_key, e)
            ciphertext = bfv_rqi.PubKeyEncrypt(
                public_key_rqi, message, (e0, e1), u, self.crt_moduli.q
            )
            c0_rqis.append(ciphertext[0])
            c1_rqis.append(ciphertext[1])

        # Recover ciphertext in Rq
        c0 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c0_rqis, self.n, self.crt_moduli
        )
        c1 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c1_rqis, self.n, self.crt_moduli
        )

        # Perform decryption in Rq basis
        dec = bfv_rq.PubKeyDecrypt(secret_key, (c0, c1), (e0, e1), e, u)

        # Assert that the two decryptions are the same
        self.assertEqual(dec.coefficients, message.coefficients)

    # Same as before, but now using secret key encryption
    def test_valid_secret_key_decryption(self):
        bfv_rq = BFV(RLWE(self.n, self.crt_moduli.q, self.t, self.discrete_gaussian))

        secret_key = bfv_rq.SecretKeyGen()
        e = bfv_rq.rlwe.SampleFromErrorDistribution()
        message = bfv_rq.rlwe.Rt.sample_polynomial()

        # Perform secret key encryption in CRT basis
        c0_rqis = []
        c1_rqis = []

        for i in range(len(self.crt_moduli.qis)):
            bfv_rqi = BFV(
                RLWE(self.n, self.crt_moduli.qis[i], self.t, self.discrete_gaussian)
            )
            ciphertext = bfv_rqi.SecretKeyEncrypt(
                secret_key, message, e, self.crt_moduli.q
            )
            c0_rqis.append(ciphertext[0])
            c1_rqis.append(ciphertext[1])

        # Recover ciphertext in Rq
        c0 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c0_rqis, self.n, self.crt_moduli
        )
        c1 = CRTPolynomial.from_rqi_polynomials_to_rq_polynomial(
            c1_rqis, self.n, self.crt_moduli
        )

        # Perform decryption in Rq basis
        dec = bfv_rq.SecretKeyDecrypt(secret_key, (c0, c1), e)

        # Assert that the two decryptions are the same
        self.assertEqual(dec.coefficients, message.coefficients)
