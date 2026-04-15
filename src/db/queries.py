import pandas as pd
from src.db.connection import get_connection


def get_risk_class(score: float) -> str:
    if score < 0.30:
        return "LOW"
    elif score < 0.60:
        return "MEDIUM"
    return "HIGH"


def insert_predictions(submission: pd.DataFrame, model_version: str, table_name: str = "predictions_log"):
    """
    Insert predictions into PostgreSQL.

    Expected submission columns:
    - SK_ID_CURR
    - TARGET
    """
    if submission.empty:
        print("DataFrame vide, rien à insérer.")
        return

    required_cols = {"SK_ID_CURR", "TARGET"}
    missing_cols = required_cols - set(submission.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans submission: {missing_cols}")

    df = submission.copy()

    # Add extra columns for DB
    df["prediction_score"] = df["TARGET"].astype(float)
    df["risk_class"] = df["prediction_score"].apply(get_risk_class)
    df["model_version"] = model_version

    # Keep only DB columns
    df_to_insert = df[["SK_ID_CURR", "prediction_score", "risk_class", "model_version"]].copy()

    # Convert NaN / pd.NA -> None for PostgreSQL
    df_to_insert = df_to_insert.where(pd.notnull(df_to_insert), None)

    query = f"""
        INSERT INTO "{table_name}" ("SK_ID_CURR", prediction_score, risk_class, model_version)
        VALUES (%s, %s, %s, %s);
    """

    values = [tuple(row) for row in df_to_insert.to_numpy()]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, values)
        conn.commit()

    print(f"{len(values)} lignes insérées dans '{table_name}'.")


def get_training_data():
    pass


def get_client(client_id):
    pass