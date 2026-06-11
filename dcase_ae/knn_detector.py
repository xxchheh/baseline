from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class KNNConfig:
    n_neighbors: int = 5
    pca_dim: int = 64


class KNNAnomalyDetector:
    def __init__(
        self,
        normal_embeddings: np.ndarray,
        n_neighbors: int = 5,
        scaler: StandardScaler | None = None,
        pca: PCA | None = None,
    ):
        self.normal_embeddings = np.asarray(normal_embeddings, dtype=np.float32)
        if self.normal_embeddings.ndim != 2:
            raise ValueError("normal_embeddings must be a 2D array.")
        if self.normal_embeddings.shape[0] < 2:
            raise ValueError("KNN detector needs at least 2 normal embeddings.")
        self.n_neighbors = int(n_neighbors)
        self.scaler = scaler
        self.pca = pca

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        n_neighbors: int = 5,
        pca_dim: int = 64,
    ) -> "KNNAnomalyDetector":
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")
        if embeddings.shape[0] < 2:
            raise ValueError("KNN detector needs at least 2 training embeddings.")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)

        max_pca_dim = min(scaled.shape[0] - 1, scaled.shape[1])
        effective_pca_dim = min(int(pca_dim), max_pca_dim)
        pca = None
        transformed = scaled
        if effective_pca_dim >= 1 and effective_pca_dim < scaled.shape[1]:
            pca = PCA(n_components=effective_pca_dim, random_state=0)
            transformed = pca.fit_transform(scaled)

        return cls(
            normal_embeddings=transformed.astype(np.float32, copy=False),
            n_neighbors=n_neighbors,
            scaler=scaler,
            pca=pca,
        )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        transformed = embeddings
        if self.scaler is not None:
            transformed = self.scaler.transform(transformed)
        if self.pca is not None:
            transformed = self.pca.transform(transformed)
        return transformed.astype(np.float32, copy=False)

    def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        transformed = self.transform(embeddings)
        return knn_mean_distances(
            transformed,
            self.normal_embeddings,
            n_neighbors=self.n_neighbors,
            exclude_self=False,
        )

    def reference_scores(self) -> np.ndarray:
        return knn_mean_distances(
            self.normal_embeddings,
            self.normal_embeddings,
            n_neighbors=self.n_neighbors,
            exclude_self=True,
        )


def knn_mean_distances(
    query_embeddings: np.ndarray,
    normal_embeddings: np.ndarray,
    n_neighbors: int,
    exclude_self: bool = False,
) -> np.ndarray:
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    normal_embeddings = np.asarray(normal_embeddings, dtype=np.float32)
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.reshape(1, -1)
    if normal_embeddings.ndim != 2:
        raise ValueError("normal_embeddings must be a 2D array.")

    distances = _euclidean_distances(query_embeddings, normal_embeddings)
    neighbor_count = int(n_neighbors) + (1 if exclude_self else 0)
    neighbor_count = min(neighbor_count, normal_embeddings.shape[0])
    nearest = np.partition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
    nearest.sort(axis=1)
    if exclude_self:
        nearest = nearest[:, 1:]
    if nearest.shape[1] == 0:
        raise ValueError("No neighbors remain after excluding self.")
    return nearest.mean(axis=1).astype(np.float32, copy=False)


def _euclidean_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.sum(a * a, axis=1, keepdims=True)
    b_norm = np.sum(b * b, axis=1, keepdims=True).T
    squared = np.maximum(a_norm + b_norm - 2.0 * np.matmul(a, b.T), 0.0)
    return np.sqrt(squared).astype(np.float32, copy=False)
