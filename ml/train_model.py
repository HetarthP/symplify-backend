import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# ✅ Load the dataset
df = pd.read_csv("ml/Training.csv")

# ✅ Extract target and symptoms
diseases = df.iloc[:, 0]
symptom_data = df.iloc[:, 1:]

# ✅ Get list of all possible symptoms (non-null, unique)
all_symptoms = pd.unique(symptom_data.values.ravel())
all_symptoms = sorted([s for s in all_symptoms if pd.notna(s)])

# ✅ Create binary matrix
binary_matrix = []
for _, row in symptom_data.iterrows():
    symptoms = set(row.dropna())
    binary_matrix.append([1 if symptom in symptoms else 0 for symptom in all_symptoms])

# ✅ Train model
model = RandomForestClassifier()
model.fit(binary_matrix, diseases)

# ✅ Save model and symptom list
joblib.dump(model, "ml/model.pkl")
joblib.dump(all_symptoms, "ml/symptom_list.pkl")

print("✅ Model training complete.")
