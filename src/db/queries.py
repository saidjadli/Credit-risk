import pandas as pd
from src.db.connection import get_connection


def insert_data(df: pd.DataFrame, table_name: str = "features_table"):
    if df.empty:
        print("DataFrame vide, rien à insérer.")
        return

    # Convertir NaN / pd.NA → None (important pour PostgreSQL)
    df = df.where(pd.notnull(df), None)

    columns = list(df.columns)

    # Colonnes SQL
    cols_str = ", ".join([f'"{col}"' for col in columns])

    # Placeholders
    placeholders = ", ".join(["%s"] * len(columns))

    query = f"""
        INSERT INTO "{table_name}" ({cols_str})
        VALUES ({placeholders})
        ON CONFLICT ("SK_ID_CURR") DO NOTHING;
    """

    # Convertir en liste de tuples
    values = [tuple(row) for row in df.to_numpy()]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, values)
        conn.commit()

    print(f"{len(values)} lignes insérées dans '{table_name}'.")



def get_training_data():
    pass

def get_client(id):
    pass