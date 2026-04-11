import pandas as pd
import re
from model import train_model

def sanitize_feature_names(df):
    df = df.copy()
    df.columns = [
        re.sub(r'[^A-Za-z0-9_]+', '_', str(col)).strip('_')
        for col in df.columns
    ]
    return df

def cast_object_to_category(df):
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype("category")
    return df

# Chargement
df_train = pd.read_csv('../notebooks/df_train_final.csv', header=0)
df_test = pd.read_csv('../notebooks/df_test_final.csv', header=0)

# Sauvegarder les IDs avant toute modification
t_df_id = df_test['SK_ID_CURR'].copy()

# Préprocessing
df_train = sanitize_feature_names(cast_object_to_category(df_train))
df_test = sanitize_feature_names(cast_object_to_category(df_test))

# Entraînement
results = train_model(df_train, df_test)

# Selon ton implémentation, il faut récupérer le modèle entraîné.
# Cas le plus probable : train_model retourne un dict contenant le modèle.
model = results["model"]

# Construire X_test
# On enlève  SK_ID_CURR
X_test = df_test.drop(columns=['SK_ID_CURR'])


# Prédiction des probabilités de la classe positive
t_pred = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "SK_ID_CURR": t_df_id.values,
    "TARGET": t_pred
})

submission = submission.groupby("SK_ID_CURR", as_index=False)["TARGET"].mean()

# Sauvegarde
submission.to_csv("../data/submission.csv", index=False)

print(submission.head())
print("Fichier sauvegardé : ../data/submission.csv")