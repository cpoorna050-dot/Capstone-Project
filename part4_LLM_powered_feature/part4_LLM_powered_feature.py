import os
import re
import json
import joblib
import jsonschema
import requests
import time
import pandas as pd

RANDOM_STATE = 42
SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# LLM API CONFIG — key read from environment, NEVER hardcoded
# ---------------------------------------------------------------------------

def get_llm_api_key():
    for key_name in ("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(key_name)
        if value:
            return value
    return None


def get_llm_model():
    return os.environ.get("LLM_MODEL") or "openai/gpt-4o-mini"


LLM_API_KEY = get_llm_api_key()
LLM_API_URL = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_MODEL = get_llm_model()


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("="*90)

# ---------------------------------------------------------------------------
# TASK: Set up the LLM API connection
# ---------------------------------------------------------------------------
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512, max_retries=3):
    if not LLM_API_KEY:
        print("No LLM API key found in environment variables (checked LLM_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY).")
        return None

    model_candidates = []
    if LLM_MODEL:
        model_candidates.append(LLM_MODEL)
    if LLM_MODEL != "openai/gpt-4o-mini":
        model_candidates.append("openai/gpt-4o-mini")
    if "meta-llama/llama-3.1-8b-instruct:free" not in model_candidates:
        model_candidates.append("meta-llama/llama-3.1-8b-instruct:free")

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "capstone-attrition-demo",
    }

    for model_name in model_candidates:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=60)
            except requests.RequestException as exc:
                print(f"LLM request exception: {exc}")
                return None

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

            if response.status_code == 429 and attempt < max_retries - 1:
                print(f"Rate limited on {model_name} — waiting 8 seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(8)
                continue

            if response.status_code == 429 and model_name != model_candidates[-1]:
                print(f"{model_name} is rate limited; trying fallback model...")
                break

            print(f"LLM API call failed - status code: {response.status_code}")
            print(response.text[:500])
            return None

    return None

# ---------------------------------------------------------------------------
# TASK: PII guardrail
# ---------------------------------------------------------------------------
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern,text) or re.search(phone_pattern,text))

def call_llm_guarded(system_prompt,user_prompt,temperature=0.0,max_tokens=512):
    if has_pii(user_prompt):
        print("Input blocked: PII detected")
        return None
    return call_llm(system_prompt,user_prompt,temperature=temperature,max_tokens=max_tokens)

# ---------------------------------------------------------------------------
# TASK: JSON schema for the explanation output (5 required scalar fields)
# ---------------------------------------------------------------------------
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties":{
        "prediction_label": {"type":"string"},
        "confidence_level": {"type":"string","enum":["low","medium","high"]},
        "top_reason": {"type":"string"},
        "second_reason": {"type":"string"},
        "next_step": {"type":"string"},
    },
    "required": [
        "prediction_label","confidence_level","top_reason",
        "second_reason","next_step"
    ],
}

FALLBACK_EXPLANATION = {
    "prediction_label" : None,
    "confidence_level" : None,
    "top_reason": None,
    "second_reason": None,
    "next_step": None,
}

def parse_and_validate(raw_response):
    if raw_response is None:
        return dict(FALLBACK_EXPLANATION), "Fail"

    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON Decode error : {e}")
        return dict(FALLBACK_EXPLANATION), "Fail"

    try:
        jsonschema.validate(instance=parsed, schema=EXPLANATION_SCHEMA)
    except jsonschema.ValidationError as e:
        print(f"Schema Validation Error:{e.message}")
        return dict(FALLBACK_EXPLANATION), "Fail"

    return parsed, "Pass"

# ---------------------------------------------------------------------------
# TASK: encode_record — preprocess a raw feature dict the same way Parts 2/3 did
# ---------------------------------------------------------------------------
def encode_record(feature_dict, reference_columns):

    row = pd.DataFrame([feature_dict])

    travel_order = {"Non-Travel": 0, "Travel_Frequently": 1, "Travel_Rarely": 2}
    if "BusinessTravel" in row.columns:
        row["BusinessTravel"] = row["BusinessTravel"].map(travel_order)
    elif "Business_Travel" in row.columns:
        row["Business_Travel"] = row["Business_Travel"].map(travel_order)

    if "OverTime" in row.columns:
        row["OverTime"] = row["OverTime"].map({"No": 0, "Yes": 1})

    nominal_columns = [c for c in ["Department", "EducationField", "Gender",
                                   "JobRole", "MaritalStatus", "Over18"]
                       if c in row.columns]

    row = pd.get_dummies(row, columns=nominal_columns, drop_first=True)

    row = row.reindex(columns=reference_columns, fill_value=0)
    return row

# ---------------------------------------------------------------------------
# SETUP: locate cleaned_data.csv (to rebuild reference columns) and best_model.pkl
# ---------------------------------------------------------------------------

section("SETUP: locate cleaned_data.csv (to rebuild reference columns) and best_model.pkl")

MODEL_SEARCH_PATH = [
    os.path.join(SCRIP_DIR,"best_model.pk1"),
    os.path.join(SCRIP_DIR,"..","part3_advanced_modeling","best_model.pk1"),
]

model_path = next((p for p in MODEL_SEARCH_PATH if os.path.exists(p)),None)
if model_path is None:
    raise FileNotFoundError(
        f" could not find best_model.pk1 in any of {MODEL_SEARCH_PATH}."
        "make sure part 3 has been run and best_model.pk1 exists."
    )

best_model = joblib.load(model_path)
print(f"Loaded Model from: {model_path}")

CSV_SEARCH_PATH = [
    os.path.join(SCRIP_DIR,"Cleaned_data.CSV"),
    os.path.join(SCRIP_DIR,"..","part1_data_prep_eda","Cleaned_data.CSV"),
]

csv_path = next((p for p in CSV_SEARCH_PATH if os.path.exists(p)),None)
if csv_path is None:
    raise FileNotFoundError(
        f"Could not find Cleaned_data.CSV in any of {csv_path}"
        "Make use part 1 has been run and Cleaned_data exists"
    )

df_ref = pd.read_csv(csv_path)
X_ref = df_ref.drop(columns=["MonthlyIncome","Attrition"])

travel_order = {"Non-Travel": 0, "Travel_Frequently": 1, "Travel_Rarely": 2}
if "BusinessTravel" in X_ref.columns:
    X_ref["BusinessTravel"] = X_ref["BusinessTravel"].map(travel_order)
elif "Business_Travel" in X_ref.columns:
    X_ref["Business_Travel"] = X_ref["Business_Travel"].map(travel_order)

if "OverTime" in X_ref.columns:
    X_ref["OverTime"] = X_ref["OverTime"].map({"No": 0, "Yes": 1})

nominal_cols = [c for c in ["Department", "EducationField", "Gender",
                            "MaritalStatus", "JobRole", "Over18"]
                if c in X_ref.columns]

X_ref = pd.get_dummies(X_ref, columns=nominal_cols, drop_first=True)

if hasattr(best_model, "feature_names_in_"):
    REFERENCE_COLUMNS = list(best_model.feature_names_in_)
else:
    REFERENCE_COLUMNS = X_ref.columns.to_list()

print(f"REFERENCE FEATURE COLUMNS: {len(REFERENCE_COLUMNS)}")

# ---------------------------------------------------------------------------
# TASK: Demonstrate call_llm with a simple test prompt
# ---------------------------------------------------------------------------
section("TEST: call_llm with a simple test prompt")
test_output = call_llm(
    system_prompt="You are helpful assistant.",
    user_prompt="Reply with only one word: hello",
    temperature=0.0,
)

print(f"Test Call Output: {test_output!r}")

# ---------------------------------------------------------------------------
# TASK: PII guardrail demonstration
# ---------------------------------------------------------------------------

section("Guardrail demonstration")

pii_input = "Please review this employee, contact them at jane.doe@example.com for follow-up."
clean_input = "Please review this employee's attrition risk based on their feature profile."

print(f"Input witn PII -> has_pii() = {has_pii(pii_input)}")
result_pii = call_llm_guarded("You are a helpful assistant.",pii_input, temperature=0.0)
print(f"Result (should be None, blocked): {result_pii}")

print(f"\nInput WITHOUT PII -> has_pii() = {has_pii(clean_input)}")
result_clean = call_llm_guarded("You are a helpful assistant.", clean_input, temperature=0.0)
print(f"Result (should proceed to LLM): {result_clean!r}")


# ---------------------------------------------------------------------------
# TASK: System prompt for structured explanation (zero-shot, temperature=0)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an HR analytics assistant that explains machine learning prediction about employee attritionr risk to non_technical HR staff

you will be given:
 - A set of employee feature values
 - The model's predicted values (0 = will not leave,1 = will leave)
 - The predicted probability attrition

your job is to produce a short, plain-languation explanation of the prediction.

Output ONLY valid JSON with exactly these 5 fields, nothing less or markdown fences:
{
"prediction_label" : "string - eg: 'Likely to stay' or 'At risk of leaving'",
"confidence_level" : "One of : low,medium,high",
"top_reason" : "string - the single most likely contributing factor, in plain language",
"second_reason" : "string - the second most likey contributing factor, in plain language",
"next_step" : "string - a brief, actional HR recommendation"
}"""


USER_PROMPT_TEMPLATE = """Employee feature values:{feature_values}
Model prediction: {predicted_class_label} (predicted class ={predicted_class})
Predicted probability of attrition: {predicted_probability:.4f}

Provide structured JSON explanation now."""

# ---------------------------------------------------------------------------
# TASK: Three hand-crafted feature-vector inputs
# ---------------------------------------------------------------------------

section("Building Three hand-crafted test employees")

test_employees= [
    {
        "Age": 24, "BusinessTravel": "Travel_Frequently", "DailyRate": 400,
        "Department": "Sales", "DistanceFromHome": 25, "Education": 2,
        "EducationField": "Marketing", "EmployeeCount": 1, "EmployeeNumber": 9001,
        "EnvironmentSatisfaction": 1, "Gender": "Male", "HourlyRate": 45,
        "JobInvolvement": 2, "JobLevel": 1, "JobRole": "Sales Representative",
        "JobSatisfaction": 1, "MaritalStatus": "Single", "MonthlyRate": 5000,
        "NumCompaniesWorked": 4, "Over18": "Y", "OverTime": "Yes",
        "PercentSalaryHike": 11, "PerformanceRating": 3,
        "RelationshipSatisfaction": 1, "StandardHours": 80, "StockOptionLevel": 0,
        "TotalWorkingYears": 1, "TrainingTimesLastYear": 0, "WorkLifeBalance": 1,
        "YearsAtCompany": 1, "YearsInCurrentRole": 0, "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0,
    },
    {
        "Age": 45, "BusinessTravel": "Non-Travel", "DailyRate": 1100,
        "Department": "Research & Development", "DistanceFromHome": 3, "Education": 4,
        "EducationField": "Life Sciences", "EmployeeCount": 1, "EmployeeNumber": 9002,
        "EnvironmentSatisfaction": 4, "Gender": "Female", "HourlyRate": 80,
        "JobInvolvement": 4, "JobLevel": 4, "JobRole": "Research Director",
        "JobSatisfaction": 4, "MaritalStatus": "Married", "MonthlyRate": 20000,
        "NumCompaniesWorked": 1, "Over18": "Y", "OverTime": "No",
        "PercentSalaryHike": 18, "PerformanceRating": 3,
        "RelationshipSatisfaction": 4, "StandardHours": 80, "StockOptionLevel": 2,
        "TotalWorkingYears": 20, "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
        "YearsAtCompany": 15, "YearsInCurrentRole": 10, "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 9,
    },
    {
        "Age": 33, "BusinessTravel": "Travel_Rarely", "DailyRate": 700,
        "Department": "Research & Development", "DistanceFromHome": 8, "Education": 3,
        "EducationField": "Technical Degree", "EmployeeCount": 1, "EmployeeNumber": 9003,
        "EnvironmentSatisfaction": 3, "Gender": "Male", "HourlyRate": 60,
        "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Laboratory Technician",
        "JobSatisfaction": 3, "MaritalStatus": "Divorced", "MonthlyRate": 12000,
        "NumCompaniesWorked": 2, "Over18": "Y", "OverTime": "No",
        "PercentSalaryHike": 14, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StandardHours": 80, "StockOptionLevel": 1,
        "TotalWorkingYears": 8, "TrainingTimesLastYear": 2, "WorkLifeBalance": 3,
        "YearsAtCompany": 6, "YearsInCurrentRole": 4, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
    },

]

for i, emp in enumerate(test_employees,1):
    print(f"Employee {i}: JobRole={emp['JobRole']}, OverTime={emp['OverTime']}, "
          f"JobSatisfaction={emp['JobSatisfaction']}")
    
# ---------------------------------------------------------------------------
# TASK: End-to-end pipeline — predict, explain, validate (temperature=0)
# ---------------------------------------------------------------------------

section("End-to-end pipeline — predict, explain, validate (temperature=0)")

demonstration_rows = []

for i,emp in enumerate(test_employees,1):
    print(f"\n--- Empoyees{i}---")
    encoded_row = encode_record(emp,REFERENCE_COLUMNS)

    Prediction_class = int(best_model.predict(encoded_row)[0])
    Prediction_probability = float(best_model.predict_proba(encoded_row)[0,1])
    Prediction_class_label = "Attrition(will leave)" if Prediction_class == 1 else "No Attrition (will not leave)"

    print(f"Predicted class: {Prediction_class} ({Prediction_class_label})")
    print(f"Predicted probability of attrition: {Prediction_probability:.4f}")
 
    feature_values_str = ", ".join(f"{k}={v}" for k, v in emp.items())
    user_prompt = USER_PROMPT_TEMPLATE.format(
        feature_values=feature_values_str,
        predicted_class_label=Prediction_class_label,
        predicted_class=Prediction_class,
        predicted_probability=Prediction_probability,
    )
 
    guardrail_blocked = has_pii(user_prompt)
    if guardrail_blocked:
        print("Guardrail: BLOCKED (PII detected)")
        raw_response = None
    else:
        print("Guardrail: PASS (no PII detected)")
        raw_response = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)
 
    print(f"Raw LLM response: {raw_response!r}")
 
    parsed_explanation, validation_status = parse_and_validate(raw_response)
    print(f"Validation status: {validation_status}")
    print(f"Parsed explanation: {parsed_explanation}")
 
    demonstration_rows.append({
        "Feature Input (summary)": f"JobRole={emp['JobRole']}, OverTime={emp['OverTime']}",
        "Predicted Class": Prediction_class,
        "Probability": round(Prediction_probability, 4),
        "Explanation JSON": json.dumps(parsed_explanation),
        "Validation Status": validation_status,
        "Guardrail": "BLOCKED" if guardrail_blocked else "PASS",
    })
 
demonstration_table = pd.DataFrame(demonstration_rows)
print("\nFull 3-row demonstration table:")
print(demonstration_table.to_string(index=False))
 
 
# ---------------------------------------------------------------------------
# TASK: Temperature A/B comparison (temp=0 vs temp=0.7)
# ---------------------------------------------------------------------------
section("TEMPERATURE A/B COMPARISON (temp=0 vs temp=0.7)")
 
temperature_comparison_rows = []
 
for i, emp in enumerate(test_employees, 1):
    encoded_row = encode_record(emp, REFERENCE_COLUMNS)
    predicted_class = int(best_model.predict(encoded_row)[0])
    predicted_probability = float(best_model.predict_proba(encoded_row)[0, 1])
    predicted_class_label = "Attrition (will leave)" if predicted_class == 1 else "No Attrition (will stay)"
 
    feature_values_str = ", ".join(f"{k}={v}" for k, v in emp.items())
    user_prompt = USER_PROMPT_TEMPLATE.format(
        feature_values=feature_values_str,
        predicted_class_label=predicted_class_label,
        predicted_class=predicted_class,
        predicted_probability=predicted_probability,
    )
 
    if has_pii(user_prompt):
        output_temp0, output_temp07 = None, None
    else:
        output_temp0 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)
        output_temp07 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7)
 
    print(f"\nEmployee {i}:")
    print(f"  temp=0.0  -> {output_temp0!r}")
    print(f"  temp=0.7  -> {output_temp07!r}")
 
    temperature_comparison_rows.append({
        "Input": f"Employee {i} ({emp['JobRole']})",
        "Output @ temp=0": output_temp0,
        "Output @ temp=0.7": output_temp07,
    })
 
temperature_comparison_table = pd.DataFrame(temperature_comparison_rows)
print("\nFull temperature comparison table:")
print(temperature_comparison_table.to_string(index=False))
