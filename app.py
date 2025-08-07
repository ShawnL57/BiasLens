import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and vectorizer
model = joblib.load("models/best_ensemble_model.pkl")
vectorizer = joblib.load("processed/vectorizer.pkl")

# App UI
st.title("🗞️ Political Bias Classifier")
st.write("Enter a news excerpt below to classify its political bias:")

user_input = st.text_area("News Excerpt", height=200)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X_input = vectorizer.transform([user_input])
        
        # Predict class label
        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Bias: **{prediction}**")

               # Predict probabilities
        try:
            # Get probabilities for each class
            probs = model.predict_proba(X_input)[0]
            labels = model.classes_ if hasattr(model, 'classes_') else ['left', 'center', 'right']

            # Create DataFrame
            prob_df = pd.DataFrame({
                'Bias': labels,
                'Probability': [f"{p*100:.2f}%" for p in probs]
            })

            st.subheader("Prediction Confidence")

            # Show table
            st.dataframe(prob_df.set_index('Bias'))

            # Convert to numeric for visualization
            prob_df['Confidence (%)'] = prob_df['Probability'].str.rstrip('%').astype(float)

            # Show bar chart
            st.bar_chart(prob_df.set_index('Bias')['Confidence (%)'])

            # Show metric for predicted bias
            predicted_bias = prob_df.loc[prob_df['Bias'] == prediction]
            confidence = float(predicted_bias['Confidence (%)'].values[0])
            st.metric(label="Confidence in Prediction", value=f"{confidence:.2f}%")

        except AttributeError:
            st.error("This model does not support probability outputs.")
