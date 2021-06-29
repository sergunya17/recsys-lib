"""
Model evaluation functions.
"""

import numpy as np


def mean_average_precision(true_interactions: np.ndarray, pred_interactions: np.ndarray) -> float:
    """Mean Average Precision score.

    Parameters
    ----------
    true_interactions : ndarray
        Two-dimensional numpy array, where `pred_interactions[user_id, item_id] = 1`
        corresponds to the true interaction.
    pred_interactions : ndarray
        Two-dimensional numpy array, where `pred_interactions[user_id]` contains a list
        of predicted `item_id`'s. The values are sorted in descending order of model
        confidence.

    Returns
    -------
    map_score : float
        Mean Average Precision score.
    """

    test_users = np.count_nonzero(true_interactions, axis=1).nonzero()[0]
    average_precisions = []

    for user in test_users:
        user_true_interactions = true_interactions[user].nonzero()[0]
        user_pred_interactions = pred_interactions[user]

        match = np.isin(user_pred_interactions, user_true_interactions)
        precision_k = np.cumsum(match) / np.arange(1, match.shape[0] + 1)
        average_precision_k = (precision_k * match).sum() / match.shape[0]

        average_precisions.append(average_precision_k)

    map_score = float(np.mean(average_precisions))

    return map_score
