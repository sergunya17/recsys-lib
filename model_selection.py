"""
Dataset splitting functions.
"""

from typing import Optional

import numpy as np


def train_test_split(interactions: np.ndarray, test_interactions_per_user: int = 10,
                     test_size: float = 0.2, random_state: Optional[int] = None):
    """Split interactions on training and test sets.

    Parameters
    ----------
    interactions : ndarray
        Two-dimensional numpy array containing user-item interactions.
    test_interactions_per_user : int, default=10
        The number of interactions that will be in the test part for each
        user from the test part. Note that the number of users who have more
        than ``test_interactions_per_user`` interactions must be at least
        ``interactions.shape[0] * test_size``.
    test_size : float, default=0.2
        The proportion of users in the test set.
    random_state : int, default=None
        Controls the randomness of the split.

    Returns
    -------
    (train, test): tuple of ndarrays
        Tuple containing train-test split of input.
    """

    if test_interactions_per_user <= 0:
        raise ValueError('Expected test_interactions_per_user > 0. '
                         f'Got {test_interactions_per_user}.')
    if not 0 < test_size < 1:
        raise ValueError(f'Expected 0 < test_size < 1. Got {test_size}.')

    rnd = np.random.RandomState(seed=random_state)

    train = interactions.copy()
    test = np.zeros(interactions.shape)

    interactions_per_user = np.count_nonzero(interactions, axis=1)
    test_candidates = (interactions_per_user > test_interactions_per_user).nonzero()[0]
    test_users_size = int(interactions.shape[0] * test_size)
    if test_users_size > test_candidates.size:
        raise ValueError('The number of users who have more than test_interactions_per_user '
                         'interactions must be at least interactions.shape[0] * test_size.')

    test_users = rnd.choice(test_candidates, size=test_users_size, replace=False)

    for user in test_users:
        user_interactions = interactions[user, :].nonzero()[0]
        test_interactions = rnd.choice(user_interactions,
                                       size=test_interactions_per_user,
                                       replace=False)

        test[user, test_interactions] = train[user, test_interactions]
        train[user, test_interactions] = 0

    return train, test
