"""Open-set model classes (kept in their own module so saved models load in the demo)."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .advanced import phone_clf, phone_vectorizer
from .config import SEED
from .experiments import make_clf
from .ship import fuse


def probs(clf, X, labels):
    p = clf.predict_proba(X)
    return p[:, [list(clf.classes_).index(l) for l in labels]]


class Router:
    """Embedding LR + phone n-gram LR, fused (weight from the shipped model), optionally with OTHER."""

    def __init__(self, w):
        self.w = w

    def fit(self, Xe, Te, y, Xe_aug=None, y_aug=None):
        self.labels = sorted(set(y))
        Xa = Xe if Xe_aug is None else np.concatenate([Xe, Xe_aug])
        ya = y if y_aug is None else np.concatenate([y, y_aug])
        self.emb = make_clf().fit(Xa, ya)
        Te = np.asarray(Te, dtype=object)
        has = Te != ""                     # clips where the phone recogniser heard nothing teach nothing
        self.vec = phone_vectorizer().fit(Te[has])
        self.pc = phone_clf().fit(self.vec.transform(Te[has]), np.asarray(y)[has])
        return self

    def proba(self, Xe, Te, labels):
        pe = probs(self.emb, Xe, labels)
        Te = np.asarray(Te, dtype=object)
        out = pe.copy()
        has = Te != ""                     # no phones -> embedding only (an empty string is not evidence)
        if has.any():
            out[has] = fuse(pe[has], probs(self.pc, self.vec.transform(Te[has]), labels), self.w)
        return out


class Distance:
    """Mahalanobis and kNN scores on PCA-reduced, standardised embeddings of known-language clips."""

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.pca = PCA(256, random_state=SEED).fit(Z)
        Z = self.pca.transform(Z)
        self.means = {c: Z[y == c].mean(0) for c in np.unique(y)}
        R = np.concatenate([Z[y == c] - m for c, m in self.means.items()])
        self.prec = LedoitWolf().fit(R).precision_
        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(Zn)
        return self

    def _z(self, X):
        return self.pca.transform(self.sc.transform(X))

    def mahalanobis(self, X):
        Z = self._z(X)
        d = np.stack([np.einsum("ij,jk,ik->i", Z - m, self.prec, Z - m) for m in self.means.values()], 1)
        return -d.min(1)

    def knn(self, X):
        Z = self._z(X)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        dist, _ = self.nn.kneighbors(Z)
        return 1 - dist.mean(1)


