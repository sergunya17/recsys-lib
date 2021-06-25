"""
Recommendation models.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class RecommendationModel(ABC):
    """An abstract base class for recommendation model"""

    @abstractmethod
    def __init__(self):

        self.train_interactions: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self):
        """Implementation of training the model should be written in this method."""

        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self):
        """Implementation of the model prediction should be written in this method."""

        raise NotImplementedError()

    def predict(self, k: int):
        """Predict k items for each user.

        Parameters
        ----------
        k : int
            The number of predicted items per user.

        Returns
        -------
        pred_interactions : ndarray
            Two-dimensional numpy array, where `matrix[user_id, item_id] = 1`
            corresponds to the predicted interaction.
        """

        pred_all_interactions = self.predict_proba()

        candidate_interactions = (1 - self.train_interactions) * pred_all_interactions
        top_k_interactions_indexes = np.argsort(candidate_interactions, axis=1)[:, ::-1][:, :k]

        pred_k_interactions = np.zeros(pred_all_interactions.shape)
        first_indexes = np.repeat(np.arange(top_k_interactions_indexes.shape[0]), k)
        second_indexes = top_k_interactions_indexes.flatten()
        pred_k_interactions[first_indexes, second_indexes] = 1

        return pred_k_interactions
