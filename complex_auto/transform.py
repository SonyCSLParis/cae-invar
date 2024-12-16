'''
Created on Sep 5, 2014

@author: Carlos E. Cancino Chacon

Austrian Research Institute for Artificial Intelligence (OFAI)
'''
import numpy as np
from sklearn.decomposition import PCA


class rPCA:
    """
    Randomized Principal Component Analysis.

    Parameters
    ----------

    data_set: array_like
        Input matrix

    M: integer
        Target dimensionality

    Methods
    -------

    x_mean
        Mean value of the data

    pca_expvar
        Explained variance

    pca_eigvec
        Eigenvectors of the Covariance matrix

    pca_coeffs
        Coefficients of the data in the PC space

    pca_transform
        Projection of the data in the M dimensional space

    """

    def __init__(self, data_set, M=2):
        self.data_set = data_set
        self.dimensionality = M
        self.rPCA = PCA(n_components=M, svd_solver='randomized')
        self.rPCA.n_components
        self.rPCA.fit(data_set)

    def pca_eigvec(self):
        """
        Principal components

        Returns
        -------
        components: numpy array
            M Principal components
        """
        return self.rPCA.components_

    def pca_expvar(self):
        """
        Percentage of variance explained by each component

        Returns
        -------

        expvar: numpy array
            Explained variance
        """
        return self.rPCA.explained_variance_ratio_

    def pca_transform(self):
        """
        Project the data in the M dimensional space

        Returns
        -------
        transform: numpy array
            Transformed data

        """
        return self.rPCA.transform(self.data_set)

    def x_mean(self):
        """
        Mean vector of the observed data

        Returns
        -------

        x_bar: numpy_array
            The mean vector of the observed data
        """
        N, D = np.shape(self.data_set)
        x_bar = np.array([])
        for oo in np.arange(0, D):
            x_bar = np.append(x_bar, np.mean(self.data_set[:, oo]))

        return x_bar

    def pca_coeffs(self, x_vector):
        """
        Transformation of an observation vector in the M principal components

        Parameters
        ----------
        x_vector: array _like
            Input vector of the same dimension as the data_set

        Returns
        -------

        x_tilde: numpy array
            Transformed vector

        """
        # Vectors as column vectors
        x = np.matrix(x_vector).T
        x_bar = np.matrix(self.x_mean()).T
        x_tilde = np.array([])
        u_vec = self.pca_eigvec()

        for oo in np.arange(0, self.dimensionality):
            u_tmp = np.matrix(u_vec[oo, :]).T
            tmp = x.T * u_tmp - x_bar.T * u_tmp
            x_tilde = np.append(x_tilde, tmp)

        return x_tilde


if __name__ == '__main__':
    pass