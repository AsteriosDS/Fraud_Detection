import streamlit as st
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# define path
path = os.path.dirname(__file__)

# Load the saved autoencoder model
model = load_model(path + "/autoencoder.h5")
model.load_weights(path + "/auto_weights.h5")

# Load test set time series data
test_data = pd.read_csv(path + "/val_st.csv")
test_data = shuffle(test_data).reset_index(drop=True)

# Streamlit app
def main():
    tab1, tab2 = st.tabs(["Fraud Detection Autoencoder", "Reconstruction Results"])
    with tab1:
        st.title("Fraud Detection Autoencoder")
        st.write("Explore time series data and classify them using the autoencoder model.")
    
        # # Create a grid layout with two columns
        # col1, col2 = st.columns((1,1), gap="large")
    
        # with col1:
        # Display micrographs of time series data in a y times z grid
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
        
        plt.tight_layout(pad=2.5)
        st.pyplot(fig)
    
        # with col2:
        # Classification functionality
        selected_timeseries = st.selectbox("Select a time series to classify", test_data.index)
        selected_data = test_data.loc[selected_timeseries]
        reconstructed = model.predict(np.array([selected_data]))[0]
        reconstruction_error = np.mean(np.abs(selected_data - reconstructed))
        
        st.subheader('Original vs Reconstructed Time Series')
        plt.figure(figsize=(10, 6))
        plt.plot(selected_data, label="Original")
        plt.plot(reconstructed, label="Reconstructed")
        plt.fill_between(range(len(selected_data)), selected_data, reconstructed, where=(reconstructed > selected_data), interpolate=True, color='green', alpha=0.5)
        plt.fill_between(range(len(selected_data)), selected_data, reconstructed, where=(reconstructed < selected_data), interpolate=True, color='red', alpha=0.5)
        plt.xlabel("Features")
        plt.ylabel("Value")
        plt.tight_layout(pad=2.5)
        
        # Add legends
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.extend([
            plt.Line2D([], [], color='green', alpha=0.5, label='Reconstructed >= Original'),
            plt.Line2D([], [], color='red', alpha=0.5, label='Reconstructed < Original')
        ])
        plt.legend(handles=handles, loc='upper right')
        st.pyplot(plt)
        
        if reconstruction_error > 0.05:
            st.warning("This time series is classified as fraud.")
        else:
            st.success("This time series is not classified as fraud.")
    with tab2:
        st.title("Reconstruction Results")
        col1, col2 =  st.columns(2)
        with col1:
            # Display the sample of 3000 time series before reconstruction
            st.subheader("Before Reconstruction")
            st.image(path + "/b4.jpg", use_column_width=True)
        with col2:
            # Display the sample of 3000 time series after reconstruction
            st.subheader("After Reconstruction")
            st.image(path + "/afterr.jpg", use_column_width=True)

if __name__ == "__main__":
    main()
