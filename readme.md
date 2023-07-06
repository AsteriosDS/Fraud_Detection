## Fraud Detection with Autoencoder Neural Net
---
### This was built with the dataset from MACHINE LEARNING GROUP - ULB [click here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
---
## Interactive dashboard: [click here](https://fraudzzzdetection.streamlit.app/)

### 1. Used 2/3 of non frauds for train/test and remaining of both classes for validation.
### 2. Autoencoder optimized with Optuna towards minimization of mse.
### 3. Optimized the threshold value K in [initial_threshold,1] where initial threshold is np.mean(reconstruction_errors) + np.std(reconstruction_errors)

### Repo contents:
#### 1. *Auto_Fraud.ipynb*: Code used to build,optimize the model and the threshold value.
#### 2. *afterr.jpg*: jpg with the reconstruction results of a sample from the dataset.
#### 3. *autoencoder.h5*,*auto_weights.h5*: Autoencoder network and its weights.
#### 4. *b4.jpg*: jpg of a sample from the dataset before reconstruction.
#### 5. *fraud_app.py*: Python file for the streamlit app.
#### 6. *requirements.txt*: Requirements for the streamlit vm.
#### 7. *val_st.csv*: csv with a sample from the validation set, used in the streamlit app.
