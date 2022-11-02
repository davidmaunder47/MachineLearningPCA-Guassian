import pytest
import numpy as np
from src.gaussian import GaussianModel
from scipy import stats
from src.stats import calculate_mean, calculate_cov


@pytest.fixture
def sample_data():
    """
    Generate a sample data matrix with n_rows = number of examples,
    n_cols = number of features
    :yields: the data matrix
    """
    x = np.random.rand(5, 3)  # You can change this sample data should you wish to
    yield x


@pytest.fixture
def mean_of_data(sample_data):
    yield np.mean(sample_data, axis=0)


def test_gaussian(sample_data):
    """
    GIVEN a sample data set
    WHEN the mean is calculated using calculate_mean
    THEN the calculated mean is close to np.mean
    :param sample_data:
    """

    """this is used to setup the variables for stats.multivariate_normal"""
    mean_test = np.mean(sample_data, axis=0)
    cov_test = np.cov(sample_data, rowvar=False)

    """this is used to setup the variables for the function me made
    we want to use the functions we made ourselves"""
    mean = calculate_mean(sample_data)
    cov = calculate_cov(sample_data, mean)

    """calling our Class we made from our gaussian model"""
    gaussian = GaussianModel(mean, cov)

    """below we will have create a random 1x3 test array"""
    test = np.random.uniform(size=(1, 3))
    "next we will set up two values and see if the difference between these two values is below 0.1 "
    value1 = stats.multivariate_normal.logpdf(test, mean_test, cov_test)
    value2 = gaussian.calculate_log_likelihood(test)

    assert abs(value2 - value1) < 0.1
