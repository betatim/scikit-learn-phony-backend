import numpy as np
from sklearn.utils.validation import check_random_state


class KMeans:
    def __init__(self, estimator):
        self.estimator = estimator
        self.n_clusters = estimator.n_clusters
        self.max_iter = estimator.max_iter
        self.tol = estimator.tol
        self.centroids = None

    def fit(self, X, y=None, sample_weight=None):
        # XXX This is just to illustrate how things might work, it
        # XXX doesn't actually implement the same behaviour as scikit-learn
        # XXX but a real backend should do that!

        random_state = check_random_state(self.estimator.random_state)

        initial_indices = random_state.permutation(len(X))[:self.n_clusters]
        self.centroids = X[initial_indices]

        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # Check for convergence
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids

        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        inertia = np.sum(min_distances**2)

        # Set the required attributes on the original estimator
        self.estimator.n_features_in_ = X.shape[1]
        self.estimator.cluster_centers_ = self.centroids
        self.estimator.labels_ = labels
        self.estimator.inertia_ = inertia
        self.estimator.n_iter_ = i

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
