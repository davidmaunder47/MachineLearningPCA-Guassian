import numpy as np


def calculate_mean(x_data):
    """
    This function uses numpy to calculate the mean vector of a dataset
    :param x_data: a 2-D numpy array with n_examples as rows and n_features as columns
    :return: mean_vec
    Note, you may not use np.mean to calculate the mean vector
    """
    r, c = x_data.shape
    sum_temp = 0
    final_array = np.empty([c])

    # we will use two for loops, here we will calculate the sum of of each
    # feature vector and then divide by the amount of each feature vector
    for i in range(c):
        for j in range(r):
            sum_temp += x_data[j, i]
        final_array[i] = sum_temp / r
        sum_temp = 0

    return final_array


def calculate_cov(x_data, mean_vec):
    """
    This function uses numpy to calculate the covariance matrix of a dataset
    :param x_data: a 2-D numpy array with n_examples as rows and n_features as columns
    :return: mean_vec
    Note, you may not use np.cov to calculate the covariance matrix
    """
    """
    r, c = x_data.shape
    deviation_array = np.subtract(x_data, np.dot(np.ones((r, r)), x_data) * (1 / r))

    return np.dot(np.transpose(deviation_array), deviation_array) """

    # initial setup
    r, c = x_data.shape
    final_array = np.empty([c, c])
    sum_temp = 0

    # We will loop through each array and we will use a bubble sort approach
    # where each feature vector will be compared to other feature vectors in our matrix
    for i in range(c):
        temp = i
        for j in range(c):
            for k in range(r):
                sum_temp += (x_data[k, temp] - mean_vec[temp]) * (x_data[k, j] - mean_vec[j])
            final_array[temp, j] = sum_temp / (r - 1)
            sum_temp = 0

    return final_array
