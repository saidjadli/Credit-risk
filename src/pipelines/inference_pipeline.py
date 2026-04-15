import os
import json
import joblib
import logging
import pandas as pd

from src.model import test_model
# from src.data.selection import drop_unused_columns
from src.config import TRAIN_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "artifacts", "model")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "..", "..", "data", "submissions")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    logging.info("Starting inference pipeline")

    test_path = os.path.join(TRAIN_DATA_DIR, "df_test_final.parquet")
    logging.info("Loading test data from %s", test_path)

    df_test = pd.read_parquet(test_path)
    logging.info("Test data loaded with shape: %s", df_test.shape)

    # logging.info("Dropping unused columns")
    # df_test = drop_unused_columns(df_test)
    # logging.info("Shape after dropping columns: %s", df_test.shape)

    model_path = os.path.join(ARTIFACTS_MODEL_DIR, "lgbm_model.pkl")
    features_path = os.path.join(ARTIFACTS_MODEL_DIR, "feature_dtypes.json")

    logging.info("Loading model from %s", model_path)
    model = joblib.load(model_path)
    logging.info("Model loaded successfully")

    logging.info("Loading feature dtypes from %s", features_path)
    feature_dtypes = load_json(features_path)
    features = list(feature_dtypes.keys())

    if "SK_ID_CURR" not in df_test.columns:
        raise ValueError("Column 'SK_ID_CURR' is missing from test data.")

    # Add missing features if needed
    missing_features = [col for col in features if col not in df_test.columns]
    extra_features = [col for col in df_test.columns if col not in features + ["SK_ID_CURR"]]

    if missing_features:
        logging.warning(
            "Missing %d features in test set. They will be added with NaN.",
            len(missing_features)
        )
        for col in missing_features:
            df_test[col] = pd.NA

    if extra_features:
        logging.info(
            "Found %d extra columns in test set. They will be ignored.",
            len(extra_features)
        )

    # Exact same order as training
    df_test_aligned = df_test[["SK_ID_CURR"] + features].copy()
    logging.info("Aligned test data shape: %s", df_test_aligned.shape)

    logging.info("Running model inference")
    submission = test_model(model, df_test_aligned)
    logging.info("Inference completed successfully")

    if isinstance(submission, pd.DataFrame):
        os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
        submission_path = os.path.join(SUBMISSIONS_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)
        logging.info("Submission saved at %s", submission_path)
    else:
        logging.warning("test_model did not return a DataFrame, so no submission file was saved.")

    logging.info("Inference pipeline finished successfully")


if __name__ == "__main__":
    main()