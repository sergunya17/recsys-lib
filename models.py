"""
Recommendation models.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Final

import numpy as np
from sklearn.metrics import pairwise_distances


class RecommendationModel(ABC):
    """An abstract base class for recommendation model."""

    @abstractmethod
    def __init__(self):

        self._train_interactions: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, train_interactions):
        """Implementation of training the model should be written in this method."""

        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self):
        """Implementation of the model prediction should be written in this method."""

        raise NotImplementedError()

    def predict(self, k_items: int) -> np.ndarray:
        """Predict k items for each user.

        Parameters
        ----------
        k_items : int
            The number of predicted items per user.

        Returns
        -------
        pred_interactions : ndarray
            Two-dimensional numpy array, where `pred_interactions[user_id]` contains a list
            of predicted `item_id`'s of length `k_items`. The values are sorted in descending
            order of model confidence.
        """

        if k_items <= 0:
            raise ValueError(f'Expected k_items > 0. Got {k_items}.')

        pred_all_interactions = self.predict_proba()

        candidate_interactions = (1 - self._train_interactions) * pred_all_interactions
        pred_interactions = np.argsort(candidate_interactions,
                                       axis=1)[:, ::-1][:, :k_items]

        return pred_interactions

    def predict_matrix(self, k_items: int) -> np.ndarray:
        """Predict k items for each user.

        Parameters
        ----------
        k_items : int
            The number of predicted items per user.

        Returns
        -------
        pred_interactions : ndarray
            Two-dimensional numpy array, where `pred_interactions[user_id, item_id] = 1`
            corresponds to the `predicted` interaction.
        """

        top_k_interactions_indexes = self.predict(k_items)

        pred_k_interactions = np.zeros(self._train_interactions.shape)
        first_indexes = np.repeat(np.arange(top_k_interactions_indexes.shape[0]), k_items)
        second_indexes = top_k_interactions_indexes.flatten()
        pred_k_interactions[first_indexes, second_indexes] = 1

        return pred_k_interactions

class KNNBasedModel(RecommendationModel):
    """KNN-based recommendation model.

    Parameters
    ----------
    k_neighbors : int, default=10
        The number of neighbors used for prediction.
    cf_type : str, default='user-based'
        Collaborative filtering type. Valid values are: ['user-based', 'item-based'].
    distance_metric : str, default='cosine'
        The metric to use when calculating distance between neighbors.
        Valid values are: ['cosine', 'euclidean', 'manhattan', 'l1', 'l2'].
    """

    _CF_TYPES: Final[List[str]] = ['user-based', 'item-based']
    _DISTANCE_METRICS: Final[List[str]] = ['cosine', 'euclidean', 'manhattan', 'l1', 'l2']

    def __init__(self, k_neighbors: int = 10, cf_type: str = 'user-based',
                 distance_metric: str = 'cosine'):

        if k_neighbors <= 0:
            raise ValueError(f'Expected k_neighbors > 0. Got {k_neighbors}.')
        if cf_type not in self._CF_TYPES:
            raise ValueError(f'Unknown CF type {cf_type}. '
                             f'Valid types are {self._CF_TYPES}.')
        if distance_metric not in self._DISTANCE_METRICS:
            raise ValueError(f'Unknown distance metric {distance_metric}. '
                             f'Valid metrics are {self._DISTANCE_METRICS}.')

        super().__init__()
        self.k_neighbors = k_neighbors
        self.cf_type = cf_type
        self.distance_metric = distance_metric
        self._distances: Optional[np.ndarray] = None

    def fit(self, train_interactions: np.ndarray):
        """Train model.

        Parameters
        ----------
        train_interactions : ndarray
            Two-dimensional numpy array, where `pred_interactions[user_id, item_id] = 1`
            corresponds to the true interaction.

        Returns
        -------
        self : returns an instance of self.
        """

        self._train_interactions = train_interactions.copy()
        if self.cf_type == 'item-based':
            self._train_interactions = self._train_interactions.T
        self._distances = pairwise_distances(self._train_interactions,
                                             metric=self.distance_metric,
                                             n_jobs=-1)
        if self.cf_type == 'item-based':
            self._train_interactions = self._train_interactions.T

        return self

    def predict_proba(self) -> np.ndarray:
        """Predict interactions probabilities.

        Returns
        -------
        pred_interactions : ndarray
            Two-dimensional numpy array, where `pred_interactions[user_id, item_id] = probability`
            corresponds to the `predicted` interaction `probability`.
        """

        if self.cf_type == 'item-based':
            self._train_interactions = self._train_interactions.T
        pred_interactions = np.zeros(self._train_interactions.shape)

        knn_indexes = np.argsort(self._distances, axis=1)[:, 1:self.k_neighbors + 1]
        for index, neighbors in enumerate(knn_indexes):
            pred_interactions[index] = self._train_interactions[neighbors].mean(axis=0)

        if self.cf_type == 'item-based':
            self._train_interactions = self._train_interactions.T
            pred_interactions = pred_interactions.T

        return pred_interactions

class SVDBasedModel(RecommendationModel):
    """SVD-based recommendation model.

    Parameters
    ----------
    k : int, default=10
        The number of singular values and vectors to compute.
        Must be 1 <= k < min(train_interactions.shape).
    """

    def __init__(self, k: int = 10):

        if k <= 0:
            raise ValueError(f'Expected k > 0. Got {k}.')

        super().__init__()
        self.k = k
        self._u: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._vt: Optional[np.ndarray] = None

    def fit(self, train_interactions: np.ndarray):
        """Train model.

        Parameters
        ----------
        train_interactions : ndarray
            Two-dimensional numpy array, where `pred_interactions[user_id, item_id] = 1`
            corresponds to the true interaction.

        Returns
        -------
        self : returns an instance of self.
        """

        self._train_interactions = train_interactions.copy()
        self._u, self._sigma, self._vt = np.linalg.svd(self._train_interactions,
                                                       full_matrices=False)
        self._sigma = np.diag(self._sigma)

        return self

    def predict_proba(self) -> np.ndarray:
        """Predict interactions probabilities.

        Returns
        -------
        pred_interactions : ndarray
            Two-dimensional numpy array, where `pred_interactions[user_id, item_id] = probability`
            corresponds to the `predicted` interaction `probability`.
        """

        sigma_k = self._sigma[0:self.k, 0:self.k]
        u_k = self._u[:, 0:self.k]
        vt_k = self._vt[0:self.k, :]

        pred_interactions = np.dot(np.dot(u_k, sigma_k), vt_k)

        pred_interactions -= pred_interactions.min()
        pred_interactions /= pred_interactions.max()

        return pred_interactions
