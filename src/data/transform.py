import os
import logging
import pandas as pd

from src.data.features import (
    transform_application_table,
    transform_bureau_tables,
    transform_previous_and_pos_cash,
    transform_credit_card_balance,
    transform_installments_payments,
    clean_feature_names
)
from src.config import ARCHIVE_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_raw_tables():
    logging.info("Reading datasets...")

    data = {
        "application_train": pd.read_csv(os.path.join(ARCHIVE_DATA_DIR, "application_train.csv")),
        "application_test": pd.read_csv(os.path.join(ARCHIVE_DATA_DIR, "application_test.csv")),
        "bureau": pd.read_csv(os.path.join(ARCHIVE_DATA_DIR, "bureau.csv")),
        "bureau_balance": pd.read_csv(os.path.join(ARCHIVE_DATA_DIR, "bureau_balance.csv")),
        "previous_application": pd.read_csv(os.path.join(ARCHIVE_DATA_DIR, "previous_application.csv")),
        "pos_cash": pd.read_csv(os.path.join(ARCHIVE_DATA_DIR, "POS_CASH_balance.csv")),
        "credit_card": pd.read_csv(os.path.join(ARCHIVE_DATA_DIR, "credit_card_balance.csv")),
        "installments": pd.read_csv(os.path.join(ARCHIVE_DATA_DIR, "installments_payments.csv")),
    }

    logging.info("Datasets loaded")
    return data


def build_final_datasets():
    data = load_raw_tables()

    logging.info("Transforming application tables...")
    df_train_tf = transform_application_table(data["application_train"])
    df_test_tf = transform_application_table(data["application_test"])

    logging.info("Transforming bureau tables...")
    bureau_tf = transform_bureau_tables(data["bureau"], data["bureau_balance"])

    logging.info("Transforming previous applications and POS cash...")
    previous_tf = transform_previous_and_pos_cash(
        data["previous_application"],
        data["pos_cash"]
    )

    logging.info("Transforming credit card balance...")
    credit_card_tf = transform_credit_card_balance(data["credit_card"])

    logging.info("Transforming installments payments...")
    installments_tf = transform_installments_payments(data["installments"])

    logging.info("Merging datasets...")
    train_df = (
        df_train_tf
        .merge(bureau_tf, on="SK_ID_CURR", how="left")
        .merge(previous_tf, on="SK_ID_CURR", how="left")
        .merge(credit_card_tf, on="SK_ID_CURR", how="left")
        .merge(installments_tf, on="SK_ID_CURR", how="left")
    )

    test_df = (
        df_test_tf
        .merge(bureau_tf, on="SK_ID_CURR", how="left")
        .merge(previous_tf, on="SK_ID_CURR", how="left")
        .merge(credit_card_tf, on="SK_ID_CURR", how="left")
        .merge(installments_tf, on="SK_ID_CURR", how="left")
    )

    train_df = clean_feature_names(train_df)
    test_df = clean_feature_names(test_df)

    return train_df, test_df