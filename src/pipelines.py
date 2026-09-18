import numpy as np
from typing import Optional
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer, OneHotEncoder
from category_encoders import BinaryEncoder
from sklearn.feature_extraction.text import CountVectorizer
from src.text_cleaner import TextCleaner
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from src.word2vec import Word2VecVectorizer



def build_sephora_pipeline(model: ClassifierMixin, vectorizer_name: str, feature_selection: Optional[str] = None) -> Pipeline:
    """
    Builds a scikit-learn pipeline for classification on the Sephora product review dataset.

    This pipeline combines text preprocessing, numerical feature transformations, 
    and optional feature selection before applying the classification model.

    The pipeline includes:
    - Text preprocessing using a custom TextCleaner and one of: BoW, TF-IDF, or Word2Vec
    - Numerical transformations: log-transform, robust scaling, binary and one-hot encoding
    - Feature union using ColumnTransformer for text and structured features
    - Optional feature selection (SelectKBest) or dimensionality reduction (PCA)
    - A classification model

    Args:
        model (ClassifierMixin): A scikit-learn compatible classifier (e.g. LogisticRegression, RandomForestClassifier).
        vectorizer_name (str): One of:
            - "bow": Bag-of-Words using CountVectorizer
            - "tf-idf": TF-IDF vectorization
            - "word2vec": Custom Word2VecVectorizer
        feature_selection (str, optional): Feature reduction method:
            - "kbest": SelectKBest with ANOVA F-value
            - "pca": PCA with 35 components
            - None (default): No dimensionality reduction

    Returns:
        Pipeline: A scikit-learn Pipeline with text and numeric preprocessing, optional feature selection, and classifier.

    Raises:
        ValueError: If `vectorizer_name` is not one of ["bow", "tf-idf", "word2vec"].

    Notes:
        - Categorical features are split into one-hot and binary encoded sets.
        - Logarithmic transformation is applied to skewed numerical features.
        - Uses ColumnTransformer to merge text and structured data streams.
        - Input data must contain a column named "combined_text" with raw text.

    Example:
        >>> model = RandomForestClassifier()
        >>> pipeline = build_sephora_pipeline(model, vectorizer_name="tf-idf", feature_selection="pca")
        >>> pipeline.fit(X_train, y_train)
        >>> y_pred = pipeline.predict(X_test)
    """
    if vectorizer_name == "bow":
        vectorizer = CountVectorizer(max_features=50)
    elif vectorizer_name == "tf-idf":
        vectorizer = TfidfVectorizer(max_features=50)
    elif vectorizer_name == "word2vec":
        vectorizer = Word2VecVectorizer()
    else:
        raise ValueError("Wrong vectorizer name!")

    categorical_onehot = ["eye_color", "skin_type", "hair_color", "variation_type"]
    categorical_binary = [
        "skin_tone",
        "secondary_category",
        "tertiary_category",
        "brand_id",
        "product_id",
    ]
    robust_transf = [
        "helpfulness",
        "price_usd",
        "loves_count",
        "rating",
        "reviews",
        "child_count",
    ]
    log_transf = [
        "total_feedback_count",
        "total_pos_feedback_count",
        "total_neg_feedback_count",
    ]

    # text data pipeline
    text_pipeline = Pipeline(
        [("clean_text", TextCleaner(batch_size=500)), ("vectorizer", vectorizer)]
    )

    # preprocesor for numerical data
    preprocessor = ColumnTransformer(
        [
            ("robust", RobustScaler(), robust_transf),
            ("log", FunctionTransformer(np.log1p, validate=False), log_transf),
            (
                "binary",
                BinaryEncoder(cols=categorical_binary, drop_invariant=True),
                categorical_binary,
            ),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_onehot,
            ),
        ],
        remainder="drop",
    )

    # pipelines concatenation
    combined_features = ColumnTransformer(
        [
            ("text", text_pipeline, "combined_text"),
            (
                "num",
                preprocessor,
                [
                    "helpfulness",
                    "price_usd",
                    "loves_count",
                    "rating",
                    "reviews",
                    "child_count",
                    "eye_color",
                    "skin_type",
                    "hair_color",
                    "variation_type",
                    "skin_tone",
                    "secondary_category",
                    "tertiary_category",
                    "brand_id",
                    "product_id",
                    "total_feedback_count",
                    "total_pos_feedback_count",
                    "total_neg_feedback_count",
                ],
            ),
        ]
    )

    # Final pipeline
    pipeline_elements = [("combined_features", combined_features), ("clf", model)]

    if feature_selection == "kbest":
        print("kbest")
        pipeline_elements.insert(
            1, ("feature_selection", SelectKBest(score_func=f_classif, k=35))
        )
    elif feature_selection == "pca":
        print("pca")
        pipeline_elements.insert(1, ("pca", PCA(n_components=35)))
    else:
        pass

    pipeline = Pipeline(pipeline_elements)
    return pipeline


def build_movies_pipeline(model: ClassifierMixin, vectorizer_name: str, feature_selection: Optional[str] = None) -> Pipeline:
    """
    Builds a scikit-learn pipeline for text classification on a movie review dataset.

    The pipeline includes:
    - Text cleaning using a custom TextCleaner transformer
    - Text vectorization using one of: Bag-of-Words (BoW), TF-IDF, or Word2Vec
    - Optional feature selection (SelectKBest) or dimensionality reduction (PCA)
    - A classification model

    Args:
        model (ClassifierMixin): A scikit-learn compatible classifier (e.g., LogisticRegression, RandomForestClassifier).
        vectorizer_name (str): Name of the vectorization method to use. One of:
            - "bow": Bag-of-Words (CountVectorizer)
            - "tf-idf": TF-IDF vectorization (TfidfVectorizer)
            - "word2vec": Word2Vec-based custom vectorizer (Word2VecVectorizer)
        feature_selection (str, optional): Optional feature reduction method:
            - "kbest": Applies SelectKBest with ANOVA F-value
            - "pca": Applies PCA with 35 components
            - None (default): No feature selection

    Returns:
        Pipeline: A scikit-learn Pipeline object with the specified preprocessing steps and classifier.

    Raises:
        ValueError: If `vectorizer_name` is not one of the allowed options.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> pipeline = build_movies_pipeline(LogisticRegression(), vectorizer_name="tf-idf", feature_selection="kbest")
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """
    if vectorizer_name == "bow":
        vectorizer = CountVectorizer(max_features=50)
    elif vectorizer_name == "tf-idf":
        vectorizer = TfidfVectorizer(max_features=50)
    elif vectorizer_name == "word2vec":
        vectorizer = Word2VecVectorizer()
    else:
        raise ValueError("Wrong vectorizer name!")

    pipeline_elements = [
        ("clean_text", TextCleaner(batch_size=500)),
        ("vectorizer", vectorizer),
        ("clf", model),
    ]

    if feature_selection == "kbest":
        print('kbest')
        pipeline_elements.insert(
            2, ("feature_selection", SelectKBest(score_func=f_classif, k=35))
        )
    elif feature_selection == "pca":
        print('pca')
        pipeline_elements.insert(2, ("pca", PCA(n_components=35)))
    else:
        pass

    pipeline = Pipeline(pipeline_elements)
    return pipeline
