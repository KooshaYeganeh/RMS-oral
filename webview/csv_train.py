import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # To save the model

# Load dataset
data = pd.read_csv("../oral_cancer.csv")

# Encode target variable using LabelEncoder
label_encoder = LabelEncoder()

# Make sure all possible categories are in the encoder
data['result'] = label_encoder.fit_transform(data['result'])  # Map labels to numerical values

# Separate features and target
X = data.drop('result', axis=1)  # Features
y = data['result']  # Target variable

# Identify numerical and categorical columns
num_cols = ["age", "size"]
cat_cols = [col for col in X.columns if col not in num_cols]

# Preprocessing for numerical and categorical data
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # handle_unknown='ignore' for unseen categories
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with preprocessor and regressor
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42, n_estimators=100))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Save the trained pipeline to a .pkl file
joblib.dump(pipeline, "trained_model.pkl")
print("Model saved as trained_model.pkl")

# Save the label encoder to a separate file (optional, for decoding predictions)
joblib.dump(label_encoder, "label_encoder.pkl")
print("Label encoder saved as label_encoder.pkl")
