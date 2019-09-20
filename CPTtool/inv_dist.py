import numpy as np
from scipy.spatial import cKDTree


class InverseDistance:
    """
    Inverse distance interpolator
    """
    def __init__(self, nb_points=6, pwr=1, tol=1e-9):
        """
        Initialise Inverse distance interpolation

        :param nb_points: (optional) number of k-nearest neighbours used for interpolation. Default is 6
        :param pwr: (optional) power of the inverse. Default is 1
        :param tol: (optional) tolerance added to the point distance to overcome division by zero. Default is 1e-9
        """
        # define variables
        self.tree = []  # KDtree with nearest neighbors
        self.zn = []  # interpolation results
        self.training_data = []  # training data
        self.training_points = []  # training points
        # settings
        self.nb_near_points = nb_points
        self.power = pwr
        self.tol = tol
        return

    def interpolate(self, training_points, training_data):
        """
        Define the KDtree

        :param training_points: array with the training points
        :param training_data: data at the training points
        :return:
        """

        # assign to variables
        self.training_points = training_points  # training points
        self.training_data = training_data  # data at the training points

        # compute Euclidean distance from grid to training
        self.tree = cKDTree(self.training_points)

        return

    def predict(self, prediction_points):
        """
        Perform interpolation with inverse distance method

        :param prediction_points: prediction points
        :return:
        """

        # get distances and indexes of the closest nb_points
        dist, idx = self.tree.query(prediction_points, self.nb_near_points)
        dist += self.tol  # to overcome division by zero

        # compute weights
        weights = self.training_data[idx.ravel()].reshape(idx.shape)

        # interpolate
        self.zn = np.sum(weights / dist ** self.power, axis=1) / np.sum(1. / dist, axis=1)

        return
