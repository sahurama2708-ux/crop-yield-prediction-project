# Crop Yield Estimation - prediction fix

## What was corrected

- Rebuilt the model input using the feature names saved with the trained 113-input model.
- The previous code tried to write to `Area`, `Rainfall`, `Pesticides`, and `Temperature`, but the trained model expects `Year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`, `Area_*`, and `Item_*` columns.
- Maps this India-focused app to `Area_India` and maps each supported crop to the exact `Item_*` feature from training.
- Uses standardization statistics derived from the supplied raw training CSV. The supplied `scaler.pkl` is an identity scaler and does not scale raw form values.
- Removes the fake yield fallback and fixed 94.5% confidence.
- Converts the model target (`hg/ha_yield`) to quintals per acre using the correct unit conversion, and calculates total quintals using the entered acreage.
- Adds `/health`, input validation, visible frontend errors, and a Render/Gunicorn configuration.

## Important model limitations

The training feature list contains country, crop, year, rainfall, pesticide use, and temperature. It does **not** contain district, season, soil type, or farm acreage. District is therefore treated as India for the model; season and soil are retained only as history metadata; acreage is used only to calculate total estimated output from yield per acre. The model's training data ends in 2013, so the prediction uses 2013 as the latest in-range year.

The model was not retrained. The app now sends correctly named and standardized features to the existing model. A real inference check requires installing the pinned TensorFlow dependencies and running the deployed service.

## Deploy on Render

1. Replace the files in the GitHub repository with the contents of this project ZIP.
2. Commit and push to the branch connected to Render.
3. In Render, confirm the service root directory is the repository root. The included `render.yaml` starts Gunicorn from `app/`.
4. Deploy. Check `https://YOUR-SERVICE.onrender.com/health`; it should return `model_loaded: true`.
5. Test a prediction in the dashboard. If `/health` returns 503, open Render Logs and check the model-loading error.

Python version: 3.11.11.
