# model_definitions.py
# --------------------------------------------------
# Este archivo permite que Streamlit reconstruya
# el preprocessor usado al entrenar el modelo.
# --------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ----------- COLUMNAS EXACTAS USADAS ------------

num_features = [
    'Age',
    'Daily_Usage_Time (minutes)',
    'Posts_Per_Day',
    'Likes_Received_Per_Day',
    'Comments_Received_Per_Day',
    'Messages_Sent_Per_Day'
]

cat_features = [
    'Gender',
    'Platform',
    'Dominant_Emotion',
    'usage_bin'
]

# ----------- PIPELINES ------------

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

# ----------- PREPROCESSOR GLOBAL -----------

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])
