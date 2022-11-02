import math
import numpy as np


class GaussianModel:
    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov
        self.d = len(self.mean)  # Set this to the feature dimension

    def calculate_log_likelihood(self, x):
        """
        Calculate the log-likelihood for
        :param x:
        :return:
        """
        """
         This function is created by calculating three distinct parts of the MLE function
         The first part is dealing with dimensionality the second part is calculating the log of the determinant 
         and the third part is calculating the log of the multiplicative answer of " (x - u).Transpose * inverse of the 
         covariance matrix * (x - u)", where x is the input data array and u is the mean vector array.
         """

        # This is used to calculate the log based dimensionality of this function
        pi_d_part = -1 / 2 * self.d * (math.log(2 * 3.14))

        # part 2, here we will use a temp variable to make the function easier to read and calculate
        temp = np.linalg.det(self.cov)
        determinant = math.log(temp) * (-0.5)

        # part 3, here will be transform our matrix's into a 1x1 matrix
        x_less_mean = np.subtract(x, self.mean)
        x_less_mean_transpose = np.transpose(x_less_mean)
        part3 = (-1 / 2) * np.dot(np.dot(x_less_mean, np.linalg.inv(self.cov)), x_less_mean_transpose)

        return pi_d_part + determinant + part3
