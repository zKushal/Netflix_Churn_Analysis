from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="centered",
)

MODEL_PATH = Path("model/netflix_churn_model.pkl")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f4f4f4 0%, #ececec 100%);
        }
        .block-container {
            max-width: 860px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .form-shell {
            background: #ffffff;
            border-radius: 18px;
            padding: 2rem 2rem 1.5rem 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            border: 1px solid #e7e7e7;
        }
        .form-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            color: #1f2937;
            margin-bottom: 1.5rem;
        }
        .stButton > button,
        .stFormSubmitButton > button {
            background: #e50914;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            min-height: 3.2rem;
        }
        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: #c40812;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_PATH}. Run the notebook export cell first."
        )
    return joblib.load(MODEL_PATH)


def build_feature_frame(bundle: dict, user_inputs: dict) -> pd.DataFrame:
    feature_names = bundle["feature_names"]
    label_encoders = bundle["label_encoders"]

    encoded_inputs = user_inputs.copy()
    for column, encoder in label_encoders.items():
        encoded_inputs[column] = int(encoder.transform([user_inputs[column]])[0])

    return pd.DataFrame([[encoded_inputs[name] for name in feature_names]], columns=feature_names)


def parse_number(raw_value: str, field_name: str, *, integer: bool = False) -> float | int:
    value = raw_value.strip()
    if not value:
        raise ValueError(f"{field_name} is required.")

    try:
        parsed = int(value) if integer else float(value)
    except ValueError as exc:
        number_type = "a whole number" if integer else "a valid number"
        raise ValueError(f"{field_name} must be {number_type}.") from exc

    return parsed


def main() -> None:
    inject_styles()

    try:
        bundle = load_bundle()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoders = bundle["label_encoders"]
    st.markdown('<div class="form-shell">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">Netflix Churn Predictor</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        age_input = st.text_input("Age:", placeholder="Enter age")
        watch_time_input = st.text_input("Watch Time (Hours):", placeholder="Enter hours watched")
        subscription_type = st.selectbox(
            "Subscription Type:",
            options=list(label_encoders["Subscription_Type"].classes_),
            index=None,
            placeholder="-- Select --",
        )
        favorite_genre = st.selectbox(
            "Favorite Genre:",
            options=list(label_encoders["Favorite_Genre"].classes_),
            index=None,
            placeholder="-- Select --",
        )
        country = st.selectbox(
            "Country:",
            options=list(label_encoders["Country"].classes_),
            index=None,
            placeholder="-- Select --",
        )
        days_since_last_login = st.text_input(
            "Days Since Last Login:",
            placeholder="Enter days",
        )
        submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        missing_fields = []
        if subscription_type is None:
            missing_fields.append("Subscription Type")
        if favorite_genre is None:
            missing_fields.append("Favorite Genre")
        if country is None:
            missing_fields.append("Country")

        if missing_fields:
            st.error(f"Please select: {', '.join(missing_fields)}.")
            st.stop()

        try:
            age = parse_number(age_input, "Age", integer=True)
            watch_time_hours = parse_number(watch_time_input, "Watch Time (Hours)")
            inactivity_days = parse_number(days_since_last_login, "Days Since Last Login", integer=True)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        raw_inputs = {
            "Age": age,
            "Country": country,
            "Subscription_Type": subscription_type,
            "Watch_Time_Hours": watch_time_hours,
            "Favorite_Genre": favorite_genre,
        }

        features = build_feature_frame(bundle, raw_inputs)
        scaled_features = scaler.transform(features)
        churn_probability = float(model.predict_proba(scaled_features)[0, 1])
        prediction = int(model.predict(scaled_features)[0])

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("High churn risk")
        else:
            st.success("Low churn risk")

        st.metric("Churn Probability", f"{churn_probability:.2%}")
        st.progress(churn_probability)

        st.subheader("Input Summary")
        summary_inputs = raw_inputs | {"Days_Since_Last_Login": inactivity_days}
        st.dataframe(pd.DataFrame([summary_inputs]), use_container_width=True)

        st.info(
            "The form includes Days Since Last Login to match your desired UI. The current saved model still predicts using Age, Country, Subscription Type, Watch Time Hours, and Favorite Genre only."
        )


if __name__ == "__main__":
    main()