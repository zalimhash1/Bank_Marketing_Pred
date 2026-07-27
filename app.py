from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ===============================
# LOAD MODEL
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "bank_marketing_data.pkl"), "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
ordinal_encoder = saved["ordinal_encoder"]
one_hot_encoder = saved["one_hot_encoder"]
label_encoder = saved["label_encoder"]


# ===============================
# HOME
# ===============================

@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# PREDICT
# ===============================

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])

    job = request.form["job"]
    marital = request.form["marital"]

    education = request.form["education"]

    default = request.form["default"]
    housing = request.form["housing"]
    loan = request.form["loan"]

    contact = request.form["contact"]

    month = request.form["month"]
    day_of_week = request.form["day_of_week"]

    duration = int(request.form["duration"])
    campaign = int(request.form["campaign"])
    pdays = int(request.form["pdays"])
    previous = int(request.form["previous"])

    poutcome = request.form["poutcome"]

    emp_var_rate = float(request.form["emp_var_rate"])
    cons_price_idx = float(request.form["cons_price_idx"])
    cons_conf_idx = float(request.form["cons_conf_idx"])
    euribor3m = float(request.form["euribor3m"])
    nr_employed = float(request.form["nr_employed"])


    # ===============================
    # CREATE RAW DATAFRAME
    # ===============================

    df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed
    }])

    # ===============================
    # ORDINAL ENCODING
    # ===============================

    ordinal_cols = [
        "education",
        "month",
        "day_of_week",
        "poutcome"
    ]

    df[ordinal_cols] = ordinal_encoder.transform(df[ordinal_cols])

    # ===============================
    # ONE HOT ENCODING
    # ===============================

    onehot_cols = [
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "contact"
    ]

    encoded = one_hot_encoder.transform(df[onehot_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=one_hot_encoder.get_feature_names_out(onehot_cols),
        index=df.index
    )

    df = df.drop(columns=onehot_cols)

    df = pd.concat([df, encoded_df], axis=1)

    # ===============================
    # FEATURE ORDER
    # ===============================

    expected_columns = [
        'age',
        'education',
        'month',
        'day_of_week',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'poutcome',
        'emp.var.rate',
        'cons.price.idx',
        'cons.conf.idx',
        'euribor3m',
        'nr.employed',
        'job_blue-collar',
        'job_entrepreneur',
        'job_housemaid',
        'job_management',
        'job_retired',
        'job_self-employed',
        'job_services',
        'job_student',
        'job_technician',
        'job_unemployed',
        'marital_married',
        'marital_single',
        'default_unknown',
        'housing_yes',
        'loan_yes',
        'contact_telephone'
    ]

    # Add missing columns (if user selected dropped categories)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Arrange columns in same order as training
    df = df[expected_columns]


    # ===============================
    # PREDICTION
    # ===============================

    prediction = model.predict(df)[0]


    # ===============================
    # CONFIDENCE
    # ===============================

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df)[0]
        score = round(max(probability) * 100, 2)
    else:
        score = 95


    # ===============================
    # RESULT
    # ===============================

    prediction = label_encoder.inverse_transform([prediction])[0]

    if prediction == "no":
        result = "Customer Will Subscribe"
        color = "#22c55e"
    else:
        result = "Customer Will NOT Subscribe"
        color = "#ef4444"

    angle = score * 3.6


    # ===============================
    # RETURN
    # ===============================

    return render_template(
        "index.html",
        prediction=result,
        score=score,
        color=color,
        angle=angle
    )

if __name__ == "__main__":
    app.run(debug=True)