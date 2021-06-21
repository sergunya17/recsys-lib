"""
Loaders for recsys dataset.
"""

import numpy as np
import pandas as pd


class InteractionData:
    """Data loader for user-item interactions.

    Parameters
    ----------
    file_path : str
        Path to ``.csv`` file with user-item interaction data. File should contain 3 columns
        in the following order: `user_id`, `item_id`, `value`.
    """

    def __init__(self, file_path: str):

        self.file_path = file_path

    def load(self, **kwargs) -> pd.DataFrame:
        """Load user-item interaction data as DataFrame.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to be passed to ``pd.read_csv``.

        Returns
        -------
        DataFrame
            Two-dimensional data structure with labeled axes.
        """

        interactions = pd.read_csv(self.file_path, **kwargs)
        interactions.columns = ['user_id', 'item_id', 'value']

        return interactions

    def load_matrix(self, **kwargs) -> np.ndarray:
        """Load user-item interaction data as numpy matrix.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to be passed to ``pd.read_csv``.

        Returns
        -------
        matrix : ndarray
            Two-dimensional numpy array, where `matrix[user_id, item_id] = value`.
            Other values are `zeros` by default.
        """

        interactions_df = self.load(**kwargs)

        max_user_id = interactions_df['user_id'].max()
        max_item_id = interactions_df['item_id'].max()

        interaction_matrix = np.zeros((max_user_id + 1, max_item_id + 1))
        for row in interactions_df.itertuples():
            interaction_matrix[row.user_id, row.item_id] = row.value

        return interaction_matrix
