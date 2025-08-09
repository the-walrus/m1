Heart Attack Risk Prediction

A small FastAPI web application that accepts a CSV with patient data, runs a trained pipeline, and outputs a list of IDs of people with high heart-attack risk.
Project structure

m1/<br/>
├── app.py              # FastAPI application <br/>
├── model.py            # HeartRiskModel class: load pipeline + feature engineering + predict()<br/>
├── test.py             # Simple script to check /health and send a test CSV<br/>
├── best_model.pkl      # Serialized GridSearchCV.best_estimator_<br/>
├── templates/          # HTML templates<br/>
│   ├── start_form.html # CSV upload form<br/>
│   └── res_form.html   # Display list of high-risk IDs<br/>
└── tmp/                # Temp folder for uploaded CSV files<br/>

Installation & run

    Clone (or download) the entire m1 folder.

    Switch into it:

cd m1

Install dependencies (a virtual env is recommended):

pip install fastapi uvicorn joblib pandas scikit-learn catboost jinja2

Start the server:

    python -m uvicorn app:app --reload

    Open in a browser: http://127.0.0.1:8000/

How it works
1) app.py

Initialization

    Loads the pipeline from best_model.pkl into HeartRiskModel.

    Sets a threshold threshold = 0.5.

Endpoints

    GET /health → returns 200 OK.

    GET / → renders start_form.html.

    POST /predict →

        Receives a file under the key file.

        Saves it to the temp folder tmp/.

        Passes the DataFrame to HeartRiskModel.predict().

        If preprocessing or prediction fails (e.g., missing columns), redirects to /?error=….

        Forms a list of high_risk patient IDs where model.predict_proba ≥ threshold.

        Displays this list via res_form.html.

2) model.py

Class HeartRiskModel:

TOP_FEATURES — 14 features

    Top-10 by SHAP:
    bmi, income, systolic_blood_pressure, triglycerides,
    sedentary_hours_per_day, cholesterol, exercise_hours_per_week,
    heart_rate, age, stress_level

    Plus 4 engineered:
    lifestyle_risk, active_score, pulse_pressure, age_group

Methods

    __init__(model_path, threshold) — loads the pipeline from model_path.

    _feature_engineering(df) — creates:

        lifestyle_risk = smoking + alcohol_consumption + obesity

        active_score = exercise_hours_per_week – sedentary_hours_per_day

        pulse_pressure = systolic_blood_pressure – diastolic_blood_pressure

        age_group = pd.qcut(age, 4, labels=False)

    predict(df) —

        Normalizes column names to lowercase snake_case.

        Calls _feature_engineering.

        Keeps only TOP_FEATURES and passes them through the saved pipeline.

        Returns three lists: all patient IDs, binary predictions by threshold, probabilities.

3) test.py

A simple script that verifies:

    GET http://127.0.0.1:8000/health → 200 OK

    POST http://127.0.0.1:8000/predict with any CSV → returns an HTML page with results

Model training

Data: heart_train.csv and heart_test.csv:

    Labeled train (with target), unlabeled test.

    28 columns: anthropometrics, habits, blood pressure, chronic conditions, blood biochemistry + id + target heart_attack_risk_binary.

EDA + FE

    Plots (hist, boxplot) for numerical features.

    Class balance / discreteness of binary features.

    Categorical review (gender, diet).

    Removed heart_rate outliers (> 0.2).

    Removed missing values in binary features (241 records).

Correlation

    No strong multicollinearity (max up to ±0.5).

    Weak linear relationship with the target → non-linear models are desirable.

Models

    GridSearchCV (cv=5, scoring='f1_weighted') over a pipeline:

        LogisticRegression + OneHotEncoder + passthrough (for binary + numerical)

        RandomForestClassifier(class_weight='balanced')

    Initial results: ROC-AUC ≈ 0.56, F1_weighted ≈ 0.58.

    Separately tuned CatBoostClassifier(...).fit(..., cat_features=...) → ROC-AUC ≈ 0.55.

Outcome

    Best model: RandomForestClassifier with class_weight='balanced', hyperparameters tuned by F1_weighted = 0.58 on cross-validation.
