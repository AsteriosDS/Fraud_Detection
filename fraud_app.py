import streamlit as st
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# define path
path = os.path.dirname(__file__)

# Load the saved autoencoder model
model = load_model(path + "/autoencoder.h5")
model.load_weights(path + "/auto_weights.h5")

# Streamlit app
def main():
    st.title("Fraud Detection Autoencoder")
    st.write("Explore time series data and classify them using the autoencoder model.")

    # Load test set time series data (replace with your own data)
    test_data = pd.read_csv("test_data.csv")
    samples = test_data.sample(n=500, random_state=42)

    # Display micrographs of time series data
    st.subheader("Micrographs of Time Series")
    for i, row in samples.iterrows():
        st.line_chart(row)

    # Classification functionality
    selected_timeseries = st.selectbox("Select a time series to classify", samples.index)
    selected_data = samples.loc[selected_timeseries]
    reconstructed = model.predict(np.array([selected_data]))[0]
    reconstruction_error = np.mean(np.abs(selected_data - reconstructed))
    st.subheader("Reconstructed Time Series")
    st.line_chart(reconstructed)

    if reconstruction_error > 0.05:
        st.warning("This time series is classified as fraud.")
    else:
        st.success("This time series is not classified as fraud.")

if __name__ == "__main__":
    main()
