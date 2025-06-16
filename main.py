from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

app = FastAPI()

# ✅ CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load rules from rules.json
with open("ml/rules.json", "r") as f:
    rules = json.load(f)

# ✅ Input schema
class SymptomInput(BaseModel):
    symptoms: list[str]

@app.get("/")
def read_root():
    return {"message": "MedMate backend is running."}

@app.post("/predict")
def predict_disease(data: SymptomInput):
    try:
        input_symptoms = set(s.strip().lower().replace(" ", "_") for s in data.symptoms)

        best_match = None
        best_confidence = 0

        for disease, symptom_sets in rules.items():
            for symptom_set in symptom_sets:
                known_set = set(symptom_set)
                matched = input_symptoms & known_set

                if matched:
                    confidence = len(matched) / len(known_set)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = disease

        if best_match:
            return {
                "prediction": best_match,
                "confidence": round(best_confidence * 100, 2)
            }

        return {"prediction": "No match found", "confidence": 0}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

