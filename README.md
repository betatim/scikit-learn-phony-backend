# Exploring scikit-learn backends

This repository is part of exploring how scikit-learn backends
could look like.

This backend only implements (parts of) the `KMeans` estimator.


## Installing

To install this backend checkout the repository and run
```
pip install -e.
```


## Using it

After installing the backend and the scikit-learn branch from
https://github.com/scikit-learn/scikit-learn/pull/30250

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(random_state=42)

# This won't get dispatched because this backend
# does not support the default `init` method.
km = KMeans(random_state=42)
km.fit(X, y)
assert km._backend is None

# This will get dispatched because this backend
# supports `init="random"`.
km = KMeans(init="random", random_state=42)
km.fit(X, y)
assert km._backend is not None
```