import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the data from a CSV file
df = pd.read_csv('oral_cancer.csv')

# Encode the 'result' column (target variable) and save the encoder
result_label_encoder = LabelEncoder()
df['result'] = result_label_encoder.fit_transform(df['result'])
joblib.dump(result_label_encoder, 'oral_result_label_encoder.pkl')  # Save the encoder
print("Result label encoder saved as oral_result_label_encoder.pkl")

# List of categorical columns to encode
categorical_columns = ['gender', 'location', 'color', 'surface', 'texture', 
                       'lymphnode_involvment', 'lymphnode_location', 'lymphnode_side', 
                       'lymphnode_texture', 'lymphnode_tenderness', 'lymphnode_mobility', 
                       'smoking', 'alcohol']

# Apply label encoding to categorical columns
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])  # Use separate encoder for each categorical column

# Split data into features (X) and target (y)
X = df.drop(columns=['result'])
y = df['result']

# Apply SMOTE to balance the dataset with k_neighbors set to 2
smote = SMOTE(random_state=42, k_neighbors=2)
X_res, y_res = smote.fit_resample(X, y)

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save the trained model to a file using joblib
joblib.dump(model, 'oral_cancer_ml_random_forest_model.pkl')
print("Model saved as oral_cancer_ml_random_forest_model.pkl")
