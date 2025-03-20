from itertools import chain, combinations
import pandas as pd
import numpy as np
from treeinfo import extract_tree_info
from tqdm import tqdm
from scipy.special import binom
from math import comb


def EXPVALUE(x, S, tree):
    """
    Estimates E[f(x) | x_S] using Algorithm 1.
    :param x: Input sample (1D NumPy array)
    :param S: Set of selected features
    :param tree: Dictionary containing tree information
    :return: Expected value E[f(x) | x_S]
    """

    def G(j):
        # node in the dictionary ?
        if j not in tree["v"]:
            return 0.0  # default val

        # node is a leaf? ==> return its value
        if isinstance(tree["v"][j], (int, float)):
            return float(tree["v"][j])

        # node is internal? ==> ensure the feature index exists
        if j not in tree["d"]:
            return 0.0

        # feature index and threshold
        feature_index = tree["d"][j]
        threshold = float(tree["t"][j])
        left_child = tree["a"][j]
        right_child = tree["b"][j]

        # children exist in the dictionary?
        if left_child not in tree["r"] or right_child not in tree["r"]:
            return 0.0

        cover_left = float(tree["r"][left_child])
        cover_right = float(tree["r"][right_child])
        cover_total = float(tree["r"][j])

        # feature in S? ==> follow the decision path
        if feature_index in S:
            return G(left_child) if x[feature_index] <= threshold else G(right_child)
        else:
            return (
                (G(left_child) * cover_left + G(right_child) * cover_right)
                / cover_total
                if cover_total > 0
                else 0.0
            )

    return G(0)  # Start at root node (assume it's indexed at 0)


def powerset(iterable):
    """Subsequences of the iterable from shortest to longest.
    :param iterable: Input iterable
    :return: Subsequences of the input
    """
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def compute_tree_shap(xgb_model, x, features, feature_names):
    """
    Compute SHAP values using the exact Shapley formula but summing over subsets instead of permutations.

    :param xgb_model: Trained XGBoost model
    :param x: Input sample (1D NumPy array)
    :param features: Set of feature indices to analyze
    :param feature_names: List of feature names
    :return: Dictionary with SHAP values for each feature
    """
    num_trees = xgb_model.get_booster().num_boosted_rounds()
    total_shap_value = {feat: 0.0 for feat in features}
    n_f = len(features)

    for tree_index in range(num_trees):
        tree = extract_tree_info(xgb_model, tree_index, feature_names)

        for current_feature in features:
            for subset in powerset(features - {current_feature}):
                weight = 1 / (binom(n_f - 1, len(subset)) * n_f)

                fx_S = EXPVALUE(x, set(subset), tree)
                fx_S_union_i = EXPVALUE(x, set(subset) | {current_feature}, tree)
                marginal_contribution = fx_S_union_i - fx_S

                total_shap_value[current_feature] += weight * marginal_contribution

    return {feature_names[feat]: shap for feat, shap in total_shap_value.items()}


def compute_shap_for_dataset(
    xgb_model, X_train, features, nb_samples=None, precise_samples=None
):
    """
    Compute the SHAP values for each sample in X_train.
    :param xgb_model: Trained XGBoost model
    :param X_train: Dataset (NumPy array or Pandas DataFrame)
    :param features: List of feature we want to study
    :return: SHAP values for all samples in X_train
    """
    feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None

    if isinstance(features, list) and feature_names is not None:
        features = set(feature_names.index(f) for f in features if f in feature_names)

    shap_values = []

    if precise_samples is not None:
        for i in tqdm(precise_samples):
            x = X_train.iloc[i].values
            shap_values.append(compute_tree_shap(xgb_model, x, features, feature_names))

    else:
        if nb_samples is None:
            for i in tqdm(range(len(X_train))):
                x = X_train.iloc[i].values
                shap_values.append(
                    compute_tree_shap(xgb_model, x, features, feature_names)
                )
        else:
            for i in tqdm(range(len(X_train.head(nb_samples)))):
                x = X_train.iloc[i].values
                shap_values.append(
                    compute_tree_shap(xgb_model, x, features, feature_names)
                )

    return np.array(shap_values)
