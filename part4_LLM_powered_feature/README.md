# Part 4: LLM-Powered Feature Explanation for Employee Attrition

## Chosen Track
This project implements the LLM-powered feature track for employee attrition prediction. The workflow combines a trained attrition classifier from the earlier modeling steps with a large language model that explains each prediction in plain language for HR stakeholders.

The core idea is:
1. Use the trained machine learning model to predict whether an employee is likely to leave.
2. Pass the employee’s feature values and the model output to an LLM.
3. Ask the LLM to generate a short, structured explanation that HR staff can understand.
4. Validate the LLM response against a strict JSON schema before using it.

## Prompt Design Decisions

### 1) System prompt
A zero-shot prompt was used. The model was instructed to explain the prediction for non-technical HR staff and to return only valid JSON with exactly five fields.

```text
You are an HR analytics assistant that explains machine learning prediction about employee attritionr risk to non_technical HR staff

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
}
```

### 2) Few-shot examples
No few-shot examples were used. The task was handled as a zero-shot structured generation problem to keep the prompt simple and reproducible.

### 3) Temperature choice and rationale
The main demonstration used `temperature = 0.0`.

Reason:
- Deterministic output is preferred for structured JSON generation.
- A lower temperature reduces randomness and improves consistency in field values.
- This is especially important when the response must conform to a strict schema.

## Structured Output Validation
The LLM response is validated in two steps:

1. The raw response is parsed as JSON.
2. The parsed object is checked against a JSON schema requiring exactly these five keys:
   - `prediction_label`
   - `confidence_level`
   - `top_reason`
   - `second_reason`
   - `next_step`

If parsing or validation fails, the script falls back to a null-filled explanation object instead of returning malformed output.

## Demonstration Results
The script tested three hand-crafted employee profiles and produced the following outputs.

| Employee | Summary | Predicted Class | Probability | Validation Status | Explanation Summary |
|---|---|---:|---:|---|---|
| 1 | JobRole=Sales Representative, OverTime=Yes | 1 | 0.6872 | Pass | At risk of leaving; low job satisfaction and low environment satisfaction were highlighted as main reasons. |
| 2 | JobRole=Research Director, OverTime=No | 0 | 0.0599 | Pass | Likely to stay; long tenure and high satisfaction were highlighted as main reasons. |
| 3 | JobRole=Laboratory Technician, OverTime=No | 0 | 0.0421 | Pass | Likely to stay; stable tenure and moderate satisfaction were highlighted as main reasons. |

### Verified example outputs

- Employee 1
  - Predicted class: 1 (Attrition will leave)
  - Predicted probability: 0.6872
  - Explanation: "At risk of leaving" with low job satisfaction and low environment satisfaction as the main reasons.

- Employee 2
  - Predicted class: 0 (No attrition)
  - Predicted probability: 0.0599
  - Explanation: "Likely to stay" with long tenure and high satisfaction as the dominant reasons.

- Employee 3
  - Predicted class: 0 (No attrition)
  - Predicted probability: 0.0421
  - Explanation: "Likely to stay" with stable tenure and moderate satisfaction as the main reasons.

### Temperature comparison (temperature 0.0 vs 0.7)
The same three employee prompts were run at both temperatures to compare response stability.

| Employee | temp=0.0 | temp=0.7 |
|---|---|---|
| 1 | At risk of leaving; emphasized low job satisfaction and low relationship satisfaction. | At risk of leaving; emphasized low job satisfaction and work-life balance stress. |
| 2 | Likely to stay; emphasized long tenure and high satisfaction. | Likely to stay; emphasized high job and relationship satisfaction. |
| 3 | Likely to stay; emphasized stable tenure and moderate satisfaction. | Likely to stay; emphasized stable work history and healthy work-life balance. |

## Guardrail Test Results
A simple input guardrail was implemented to block prompts containing personally identifiable information (PII).

### Test cases
- Input with PII: `Please review this employee, contact them at jane.doe@example.com for follow-up.`
  - Result: PII detected
  - LLM call blocked

- Input without PII: `Please review this employee's attrition risk based on their feature profile.`
  - Result: No PII detected
  - LLM call proceeded

## Notes
- The project uses the trained model from the earlier modeling stages and aligns the new employee input to the same feature schema before prediction.
- The current implementation is designed for explainability and human-readable HR decision support rather than fully automated decision-making.
