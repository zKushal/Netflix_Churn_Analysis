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


def main() -> None:
    st.title("Netflix Churn Prediction")
    st.caption("Deployable Streamlit app using the bundled model artifact in model/netflix_churn_model.pkl")

    try:
        bundle = load_bundle()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoders = bundle["label_encoders"]

    with st.sidebar:
        st.subheader("Model Info")
        st.write("Model: Tuned XGBoost")
        st.write("Features used: 5")
        st.write("Artifact: model/netflix_churn_model.pkl")

    st.subheader("User Profile")

    age = st.slider("Age", min_value=18, max_value=80, value=35)
    watch_time_hours = st.slider(
        "Watch Time Hours",
        min_value=0.0,
        max_value=1000.0,
        value=120.0,
        step=1.0,
    )
    country = st.selectbox("Country", options=list(label_encoders["Country"].classes_))
    subscription_type = st.selectbox(
        "Subscription Type",
        options=list(label_encoders["Subscription_Type"].classes_),
    )
    favorite_genre = st.selectbox(
        "Favorite Genre",
        options=list(label_encoders["Favorite_Genre"].classes_),
    )

    if st.button("Predict Churn Risk", type="primary"):
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
        st.dataframe(pd.DataFrame([raw_inputs]), use_container_width=True)

        st.info(
            "This app uses the saved training artifact directly. The churn label in the notebook was built from inactivity, but the deployed model predicts from demographic and usage features only."
        )


if __name__ == "__main__":
    main()