import unittest
from detpy.DETAlgs.math_functions.lehmer_mean import LehmerMean


class TestLehmerMean(unittest.TestCase):
    def setUp(self):
        self.lehmer_mean = LehmerMean()

    def test_lehmer_mean(self):
        """left list are the factors, middle is the p value, right is the expected result"""
        test_cases = [
            ([3, 6, 2, 9, 1, 7, 2], 2, 6.133),
            ([-3, -6, -2, -9, -1, -7, -2], 2, -6.133),
            ([-3, -6, -2, -9, -1, 7, 2], 2, -15.333),
            ([5], 2, 5.0),
            ([3, 6], 2, 5.0),
            ([-3, 6], 2, 15.0),
            ([3, 6], 2, 5.0),
            ([3, 6, 2, 9, 1, 7, 2], 3, 7.239),
            ([-3, -6, -2, -9, -1, -7, -2], 3, -7.239),
            ([-3, -6, -2, -9, -1, 7, 2], 3, -3.4239)
        ]

        for mut_factors, p, expected in test_cases:
            with self.subTest(mut_factors=mut_factors, p=p, expected=expected):
                result = self.lehmer_mean.evaluate(mut_factors, p)
                self.assertAlmostEqual(result, expected, places=3)


if __name__ == '__main__':
    unittest.main()
