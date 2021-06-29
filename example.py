"""
An example of using models.
"""

from data_loaders import InteractionData
from model_selection import train_test_split
from models import KNNBasedModel, SVDBasedModel
from metrics import mean_average_precision

def main():
    """Default model training."""

    interactions = InteractionData('data/interactions.csv').load_matrix()

    # reduce the amount of data for our example
    interactions = interactions[interactions.sum(axis=1) > 50, :]
    interactions = interactions[:, interactions.sum(axis=0) > 25]

    train_interactions, test_interactions = train_test_split(interactions, random_state=42)

    knn_ub = KNNBasedModel(k_neighbors=50, cf_type='user-based')
    knn_ub_pred_interactions = knn_ub.fit(train_interactions).predict(10)

    knn_ib = KNNBasedModel(k_neighbors=50, cf_type='item-based')
    knn_ib_pred_interactions = knn_ib.fit(train_interactions).predict(10)

    svd = SVDBasedModel(k=10)
    svd_pred_interactions = svd.fit(train_interactions).predict(10)

    print('KNN user-based mAP score: ', mean_average_precision(test_interactions,
                                                               knn_ub_pred_interactions))
    print('KNN item-based mAP score: ', mean_average_precision(test_interactions,
                                                               knn_ib_pred_interactions))
    print('SVD mAP score: ', mean_average_precision(test_interactions, svd_pred_interactions))


if __name__ == '__main__':
    main()
