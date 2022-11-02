from stats import calculate_mean, calculate_cov
import numpy as np


class PCA:
    def __init__(self, reduction_dimension, mean=None, cov=None):
        self.mean = None
        self.cov = None
        self.y, self.x = x_data.shape

        # This is made so we limit the dimension reduction to be less than our input array dimensions
        # it also gives the user the option of picking the dimension they want to reduce to
        if reduction_dimension >= self.x:
            self.reduction_dimension = self.x - 1
        else:
            self.reduction_dimension = reduction_dimension

    def calculate_pca(self, x_data):
        """
        Given x_data as the input data calculate
        a matrix to calculate a PCA matrix
        :return: W
        """

        # here will will change self.mean and self.cov to equal our functions from question 1
        self.mean = calculate_mean(x_data)
        self.cov = calculate_cov(x_data, self.mean)

        # eigenvectors and eigenvalues for the from the covariance matrix
        eig_val_cov, eig_vec_cov = np.linalg.eig(self.cov)

        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # we will create a temp array so we can append our new eigen values to this array
        matrix_w = np.zeros((self.x, 1))

        # this function is created so we can enter different dimensions
        for i in range(self.reduction_dimension):
            matrix_w = np.append(matrix_w, eig_pairs[i][1].reshape(self.x, 1), axis=1)

        # delete the zero column that was created from our array above
        matrix_w = np.delete(matrix_w, 0, 1)

        return x_data.dot(matrix_w)


if __name__ == '__main__':
    # this is used for temporary testing
    x_data = np.array([[1, 2, 3, 5, 10], [4, 2, 3, 2, 1], [6, 7, 10, 4, 5]])

    # first test is 2 below our original dimensionality of 5
    pca = PCA(3)
    # second test is to see if we put in a number above our dimensionality
    pca2 = PCA(6)

    # last test is to see dimensionality reduction when our input is three below our original value
    pca3 = PCA(2)
    print("3x3 dimensionality")
    print(pca.calculate_pca(x_data))
    print("")
    print("3x4 since 6 is greater than 5, thus it will be 1 less than our original value of 5")
    print(pca2.calculate_pca(x_data))
    print("3x2 dimensionality matrix")
    print("")
    print(pca3.calculate_pca(x_data))
