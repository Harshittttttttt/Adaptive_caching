from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def make_embeddings(texts, dimensions: int = 128):
    """Build local TF-IDF/SVD vectors. Fit once on the full corpus for this simulation."""
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.95, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    dim = min(dimensions, max(2, matrix.shape[0] - 1), max(2, matrix.shape[1] - 1))
    vectors = TruncatedSVD(n_components=dim, random_state=0).fit_transform(matrix)
    return normalize(vectors).astype("float32")
