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

    # Load test set time series data
    test_data = pd.read_csv(path + "/val_st.csv")

    # Display micrographs of time series data in a 5x20 grid
    st.subheader("Micrographs of Time Series")
    num_rows = 10
    num_cols = 10
    num_plots = num_rows * num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    
    for i, row in enumerate(test_data.iterrows()):
        if i >= num_plots:
            break
        
        row_index = i // num_cols
        col_index = i % num_cols
        ax = axes[row_index, col_index]
        
        ax.plot(row[1])
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")
    
    plt.tight_layout()
    st.pyplot(fig)

    # Classification functionality
    selected_timeseries = st.selectbox("Select a time series to classify", test_data.index)
    selected_data = test_data.loc[selected_timeseries]
    reconstructed = model.predict(np.array([selected_data]))[0]
    reconstruction_error = np.mean(np.abs(selected_data - reconstructed))
    
    st.subheader("Time Series Comparison")
    plt.figure(figsize=(10, 6))
    plt.plot(selected_data, label="Original")
    plt.plot(reconstructed, label="Reconstructed")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    st.pyplot(plt)
    
    if reconstruction_error > 0.05:
        st.warning("This time series is classified as fraud.")
    else:
        st.success("This time series is not classified as fraud.")

if __name__ == "__main__":
    main()
