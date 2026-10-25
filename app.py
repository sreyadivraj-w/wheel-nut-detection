import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Wheel Nut Detection System",
    page_icon="🚗",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("wheel_nut_dataset.csv")

        required = [
            "angle_change_deg",
            "magnetic_signal_change",
            "angle_change_rate",
            "condition"
        ]

        if all(col in df.columns for col in required):
            return df[required], True

    except FileNotFoundError:
        pass

    # Demonstration dataset if the original CSV is unavailable
    np.random.seed(42)
    n = 500

    normal = pd.DataFrame({
        "angle_change_deg": np.random.normal(0.5, 0.3, n // 2),
        "magnetic_signal_change": np.random.normal(0.2, 0.1, n // 2),
        "angle_change_rate": np.random.normal(0.3, 0.2, n // 2),
        "condition": ["Normal"] * (n // 2)
    })

    warning = pd.DataFrame({
        "angle_change_deg": np.random.normal(5, 1.5, n // 2),
        "magnetic_signal_change": np.random.normal(2.5, 0.7, n // 2),
        "angle_change_rate": np.random.normal(3, 1, n // 2),
        "condition": ["Warning"] * (n // 2)
    })

    df = pd.concat([normal, warning], ignore_index=True)
    return df, False


df, original_dataset_found = load_data()

features = [
    "angle_change_deg",
    "magnetic_signal_change",
    "angle_change_rate"
]

X = df[features]
y = df["condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🚗 Wheel Nut Safety")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict", "Model Performance", "Sensor Analysis", "About"]
)

st.sidebar.markdown("---")
st.sidebar.caption("VSE Machine Learning Project")

# ---------------- HOME ----------------

if page == "Home":

    st.title("🚗 Wheel Nut Loosening Detection")
    st.subheader("Low-Cost Predictive Safety System")

    st.warning(
        "DEMONSTRATION ONLY — predictions currently use simulated "
        "or previously prepared data, not live vehicle sensor readings."
    )

    st.markdown("""
    ### Project Overview

    This project explores a low-cost system to identify possible
    wheel-nut loosening at an early stage.

    A planned Hall sensor and magnet arrangement will detect relative
    movement. An ESP32 can process sensor readings and provide an
    early warning when suspicious movement is detected.

    ### How the system works

    1. **SENSE** — collect sensor readings.
    2. **PROCESS** — calculate movement-related features.
    3. **PREDICT** — use a Decision Tree machine-learning model.
    4. **ALERT** — display a Normal or Warning result.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", len(df))
    c2.metric("Model", "Decision Tree")
    c3.metric("Test Accuracy", f"{accuracy * 100:.1f}%")

    st.caption(
        "Accuracy is measured on a held-out split of the available dataset. "
        "It does not establish real-world safety performance."
    )

# ---------------- PREDICT ----------------

elif page == "Predict":

    st.title("🔍 Wheel Nut Condition Prediction")
    st.write("Enter example sensor features to demonstrate the model.")

    angle = st.number_input(
        "Angle Change (degrees)",
        min_value=0.0,
        max_value=30.0,
        value=0.5,
        step=0.1
    )

    magnetic = st.number_input(
        "Magnetic Signal Change",
        min_value=0.0,
        max_value=20.0,
        value=0.2,
        step=0.1
    )

    rate = st.number_input(
        "Angle Change Rate",
        min_value=0.0,
        max_value=20.0,
        value=0.3,
        step=0.1
    )

    if st.button("Predict Condition", type="primary"):

        sample = pd.DataFrame([{
            "angle_change_deg": angle,
            "magnetic_signal_change": magnetic,
            "angle_change_rate": rate
        }])

        prediction = model.predict(sample)[0]

        if prediction == "Warning":
            st.error("⚠️ WARNING — Model predicts an abnormal pattern.")
        else:
            st.success("✅ NORMAL — Model predicts a normal pattern.")

        st.write("**Model prediction:**", prediction)

        st.caption(
            "This is a software demonstration, not a validated diagnosis "
            "of actual wheel-nut tightness."
        )

# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Confusion Matrix")

    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Condition")
    ax.set_ylabel("Actual Condition")
    ax.set_title("Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose())

    st.info(
        "These metrics describe performance on the available dataset. "
        "Simulated data may be easier to classify than real sensor data."
    )

# ---------------- SENSOR ANALYSIS ----------------

elif page == "Sensor Analysis":

    st.title("📈 Sensor Data Analysis")

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Sensor Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(
        importance.set_index("Sensor Feature")
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Condition Distribution")
    st.bar_chart(df["condition"].value_counts())

    st.subheader("Feature Relationships")

    fig, ax = plt.subplots()

    for condition in df["condition"].unique():
        subset = df[df["condition"] == condition]
        ax.scatter(
            subset["angle_change_deg"],
            subset["magnetic_signal_change"],
            label=condition,
            alpha=0.6
        )

    ax.set_xlabel("Angle Change (degrees)")
    ax.set_ylabel("Magnetic Signal Change")
    ax.legend()
    ax.set_title("Sensor Feature Distribution")

    st.pyplot(fig)
    plt.close(fig)

# ---------------- ABOUT ----------------

elif page == "About":

    st.title("ℹ️ About the Project")

    st.markdown("""
    ### Project Stage
    Machine-learning software demonstration.

    ### Planned Hardware
    - Hall effect sensor
    - Magnet attached to a suitable rotating component
    - ESP32 microcontroller
    - Warning indicator or display

    ### Machine Learning
    A Decision Tree Classifier maps three input features to one of
    two demonstration classes: Normal or Warning.

    ### Current Limitations
    - The current demonstration uses simulated or prepared data.
    - Live Hall sensor integration is not yet implemented.
    - Real-world testing and validation are still required.
    - The model must not be used as the sole basis for vehicle safety.

    ### Future Work
    Collect real sensor readings, validate the detection method,
    and test the system under realistic wheel operating conditions.
    """)

st.markdown("---")
st.caption("Wheel Nut Loosening Detection | VSE Project")