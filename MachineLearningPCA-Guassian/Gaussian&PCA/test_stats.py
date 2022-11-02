import pytest
import numpy as np
from stats import calculate_mean, calculate_cov
from unittest import TestCase


class Test(TestCase):

    def sample_data(self):
        """
        Generate a sample data matrix with n_rows = number of examples,
        n_cols = number of features
        :yields: the data matrix
        """
        x = np.random.rand(5, 3)  # You can change this sample data should you wish to
        return x


    def mean_of_data(self,temp):
        return np.mean(temp, axis=0)

    def test_mean(self):
        """
        GIVEN a sample data set
        WHEN the mean is calculated using calculate_mean
        THEN the calculated mean is close to np.mean
        :param sample_data:
        """
        temp =self.sample_data()
        mean = calculate_mean(temp)

        assert np.allclose(mean, np.mean(temp, axis=0))

    def test_cov(self):
        """
        GIVEN a sample data set
        WHEN the covariance matrix is calculated using calculate_cov
        THEN the calculated covariance matrix is close to that given by np.cov
        :param sample_data:
        """
        temp = self.sample_data()

        cov_mat = calculate_cov(temp, calculate_mean(temp))
        cov = np.cov(temp, rowvar=False)
        assert np.allclose(cov_mat, cov)
