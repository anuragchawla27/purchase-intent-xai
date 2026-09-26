# API Reference

The prediction API is built with FastAPI and self-documents interactively at
`/docs` (Swagger UI) and `/redoc` whenever the server is running. This file
is a static reference for anyone reading the repo offline.

**Base URL (local):** `http://localhost:8000`
**Base URL (Docker Compose):** `http://localhost:8000` (mapped from the
`api` service, see `docker-compose.yml`)

---

## `GET /health`

Readiness check. Returns whether the model, preprocessor, and SHAP explainer
loaded successfully at startup.

**Request:** none

**Response — 200 OK**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Response — 503 Service Unavailable**
Returned if the model failed to load at startup (e.g. missing artifact file).

---

## `POST /predict`

Scores a single e-commerce session and returns a purchase-intent prediction
with a SHAP-based explanation.

### Request body

| Field | Type | Constraints | Description |
|---|---|---|---|
| `Administrative` | int | ≥ 0 | Administrative pages visited |
| `Administrative_Duration` | float | ≥ 0 | Seconds spent on administrative pages |
| `Informational` | int | ≥ 0 | Informational pages visited |
| `Informational_Duration` | float | ≥ 0 | Seconds spent on informational pages |
| `ProductRelated` | int | ≥ 0 | Product-related pages visited |
| `ProductRelated_Duration` | float | ≥ 0 | Seconds spent on product pages |
| `BounceRates` | float | 0–1 | Google Analytics bounce rate |
| `ExitRates` | float | 0–1 | Google Analytics exit rate |
| `PageValues` | float | ≥ 0 | Google Analytics page value (see model card — leakage caveat) |
| `SpecialDay` | float | 0–1 | Closeness to a special day |
| `Month` | string | — | e.g. `"May"` |
| `OperatingSystems` | int | — | Integer-coded category, not a quantity |
| `Browser` | int | — | Integer-coded category, not a quantity |
| `Region` | int | — | Integer-coded category, not a quantity |
| `TrafficType` | int | — | Integer-coded category, not a quantity |
| `VisitorType` | string | — | `"Returning_Visitor"`, `"New_Visitor"`, or `"Other"` |
| `Weekend` | bool | — | Whether the session occurred on a weekend |

**Example request:**
```json
{
  "Administrative": 1,
  "Administrative_Duration": 14.65,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 19,
  "ProductRelated_Duration": 283.88,
  "BounceRates": 0.008,
  "ExitRates": 0.042,
  "PageValues": 68.58,
  "SpecialDay": 0.0,
  "Month": "May",
  "OperatingSystems": 1,
  "Browser": 1,
  "Region": 1,
  "TrafficType": 1,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```

### Response — 200 OK

```json
{
  "prediction": "Purchase",
  "purchase_probability": 0.9019,
  "confidence": "High",
  "decision_threshold": 0.45,
  "top_contributing_features": [
    { "feature": "num__PageValues", "contribution": 3.4477 },
    { "feature": "num__ProductRelated", "contribution": -0.3125 },
    { "feature": "num__TotalPages", "contribution": -0.2851 },
    { "feature": "num__ProductRelated_Duration", "contribution": -0.0969 },
    { "feature": "num__ExitRates", "contribution": 0.0658 }
  ]
}
```

| Field | Description |
|---|---|
| `prediction` | `"Purchase"` or `"No Purchase"`, based on `purchase_probability` vs. `decision_threshold` |
| `purchase_probability` | Model's raw predicted probability (0–1) |
| `confidence` | `"High"` / `"Medium"` / `"Low"`, based on distance from the decision threshold |
| `decision_threshold` | The threshold currently configured (`config.yaml`, chosen via the net-value sweep — see the model card) |
| `top_contributing_features` | Top 5 features by absolute SHAP value for this specific prediction, with sign indicating direction (positive = pushes toward Purchase) |

### Response — 422 Unprocessable Entity

Returned when the request body fails validation (e.g. `BounceRates` above 1).

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "BounceRates"],
      "msg": "Input should be less than or equal to 1",
      "input": 1.5,
      "ctx": { "le": 1 }
    }
  ]
}
```

### Response — 500 Internal Server Error

Returned for unexpected failures during prediction; the error is logged
server-side and a generic message is returned to the client.

### Notes

- Unrecognized categorical values (e.g. an unseen `Month`) do **not** cause an
  error — the underlying encoder (`handle_unknown='ignore'`) treats them as
  an unknown category and still returns a valid prediction. Verified with a
  live test during development.
- The model, preprocessor, and SHAP explainer are loaded once at application
  startup, not per request — see `src/api/dependencies.py`.

---

## Interactive docs

Run the API locally or via Docker Compose, then visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

Both are auto-generated from the same Pydantic schemas documented above
(`src/api/schemas.py`), so they can never drift out of sync with this file's
field list.
