from __future__ import annotations
import numpy as np
import pandas as pd


def load_dataset(name: str, random_state: int = 0) -> pd.DataFrame:
    """Return columns row_id, text, label, n_words. Labels are oracle labels."""
    if name == "synthetic":
        rng = np.random.default_rng(random_state)
        positive = ["excellent", "warm", "moving", "enjoyable", "smart", "funny"]
        negative = ["awful", "boring", "weak", "painful", "poor", "dull"]
        neutral = ["movie", "story", "character", "scene", "screen", "ending"]
        rows = []
        for i in range(1000):
            label = int(i % 2 == 0)
            vocab = positive if label else negative
            words = list(rng.choice(vocab, size=int(rng.integers(3, 8))))
            words += list(rng.choice(neutral, size=int(rng.integers(4, 12))))
            rng.shuffle(words)
            rows.append({"row_id": f"syn-{i}", "text": " ".join(words), "label": label})
        df = pd.DataFrame(rows)
    elif name == "nltk_movie_reviews":
        try:
            import nltk
            from nltk.corpus import movie_reviews
            try:
                movie_reviews.fileids()
            except LookupError:
                nltk.download("movie_reviews", quiet=True)
            rows = []
            for fileid in movie_reviews.fileids():
                rows.append({
                    "row_id": fileid,
                    "text": " ".join(movie_reviews.words(fileid)),
                    "label": int(movie_reviews.categories(fileid)[0] == "neg")
                })
            df = pd.DataFrame(rows)
        except Exception as exc:
            raise RuntimeError("Could not load NLTK movie_reviews. Check internet access or run --dataset synthetic.") from exc
    else:
        raise ValueError(f"Unknown dataset: {name}")
    df["n_words"] = df["text"].str.split().str.len()
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
