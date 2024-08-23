# training_loop

import numpy as np
import tensorflow as tf
from data_processor import load_data
from r_vae import RVAE

# Load data
dataframe_4 = load_data()
X = dataframe_4[['GRL']].values
y = dataframe_4[['RHOB']].values

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and compile model
input_dim = X_train.shape[1]
latent_dim = 2
hidden_dims_enc = [64, 32]
hidden_dims_dec = [32, 64]

model = RVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims_enc=hidden_dims_enc, hidden_dims_dec=hidden_dims_dec)
model.compile(optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))
