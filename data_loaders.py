"""
Loaders for recsys dataset.
"""

import os
from typing import List

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

class ObjectFeaturesData:
    """Data loader for object features.

    Parameters
    ----------
    folder_path : str
        Path to folder containing ``.csv`` files. Each file must be named like
        ``<object_prefix><delimiter><feature>.csv`` and contain at least 2 columns.
        The first one is the object id, the others are the attribute columns:
        `object_id`, `feature_column_1`, ..., `feature_column_n`.
    object_prefix : str
        Filename prefix - object name. E.g., 'user' or 'item'.
    delimiter : str, default '_'
        Separator between ``<object_prefix>`` and ``<feature>``.
    """

    def __init__(self, folder_path: str, object_prefix: str,
                 delimiter: str = '_'):

        self.folder_path = folder_path
        self.object_prefix = object_prefix
        self.delimiter = delimiter

    def load(self, features: List[str], **kwargs) -> pd.DataFrame:
        """Load object features as one DataFrame.

        Parameters
        ----------
        features : list of str
            List of feature names.
        **kwargs : dict
            Additional arguments to be passed to ``pd.read_csv``.

        Returns
        -------
        DataFrame
            DataFrame contains `all` columns from files merged on `object_id`.
        """

        result_df = pd.DataFrame(columns=['id'])
        for feature in features:
            path = os.path.join(self.folder_path,
                                f'{self.object_prefix}{self.delimiter}{feature}.csv')
            feature_df = pd.read_csv(path, **kwargs)

            new_columns = [f'{feature}_{column}' for column in feature_df.columns]
            new_columns[0] = 'id'
            feature_df.columns = new_columns

            result_df = result_df.merge(feature_df, how='outer', on='id')

        result_df = result_df.rename(columns={'id': f'{self.object_prefix}_id'})

        return result_df
