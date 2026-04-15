import os
import logging
from collections import Counter
from typing import Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


DEFAULT_MODEL_PARAMS = {
    "n_estimators": 6000,
    "learning_rate": 0.0005,
    "num_leaves": 256,
    "max_depth": -1,
    "min_data_in_leaf": 120,
    "subsample": 0.6,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def evaluate_model(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "dataset",
    save_plot_path: Optional[str] = None,
) -> float:
    """
    Evaluate a trained model using ROC AUC.
    Optionally saves the ROC curve plot instead of displaying it.
    """
    y_pred = model.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, y_pred)

    logging.info("AUC on %s = %.4f", dataset_name, auc_score)

    fpr, tpr, _ = roc_curve(y, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend()
    plt.grid(True)

    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        plt.savefig(save_plot_path, bbox_inches="tight")
        logging.info("ROC curve saved to %s", save_plot_path)

    plt.close()

    return auc_score


def train_model(
    df_train: pd.DataFrame,
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    early_stopping_rounds: int = 600,
    roc_train_plot_path: Optional[str] = None,
    roc_val_plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a LightGBM model on the provided training dataframe.

    Expected columns:
    - TARGET
    - SK_ID_CURR
    - all remaining columns are used as features
    """
    required_cols = ["TARGET", "SK_ID_CURR"]
    missing_required = [col for col in required_cols if col not in df_train.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in training dataframe: {missing_required}")

    y = df_train["TARGET"]
    X = df_train.drop(columns=["TARGET", "SK_ID_CURR"])

    if y.nunique() < 2:
        raise ValueError("TARGET must contain at least two classes.")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    counter = Counter(y_train)
    if counter[1] == 0:
        raise ValueError("Positive class count is zero in training split.")

    scale_pos_weight = counter[0] / counter[1]

    final_params = DEFAULT_MODEL_PARAMS.copy()
    if model_params:
        final_params.update(model_params)

    final_params["scale_pos_weight"] = scale_pos_weight

    logging.info("Training LightGBM model...")
    logging.info("Training set shape: %s", X_train.shape)
    logging.info("Validation set shape: %s", X_valid.shape)
    logging.info("scale_pos_weight = %.4f", scale_pos_weight)

    model = lgb.LGBMClassifier(**final_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=True)],
    )

    auc_train = evaluate_model(
        model=model,
        X=X_train,
        y=y_train,
        dataset_name="train",
        save_plot_path=roc_train_plot_path,
    )

    auc_val = evaluate_model(
        model=model,
        X=X_valid,
        y=y_valid,
        dataset_name="validation",
        save_plot_path=roc_val_plot_path,
    )

    results = {
        "auc_train": auc_train,
        "auc_val": auc_val,
        "model": model,
        "features": X.columns.tolist(),
        "params": final_params,
    }

    return results


def test_model(model: lgb.LGBMClassifier, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on a test dataframe.

    Expected columns:
    - SK_ID_CURR
    - all remaining columns must match the training features
    """
    if "SK_ID_CURR" not in df_test.columns:
        raise ValueError("Column 'SK_ID_CURR' is missing from test dataframe.")

    ids = df_test["SK_ID_CURR"]
    X = df_test.drop(columns=["SK_ID_CURR"])

    logging.info("Running prediction on test set with shape: %s", X.shape)

    predictions = model.predict_proba(X)[:, 1]

    submission = pd.DataFrame(
        {
            "SK_ID_CURR": ids.values,
            "TARGET": predictions,
        }
    )

    return submission