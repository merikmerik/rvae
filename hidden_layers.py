# build hidden layers and skip connections

from tensorflow.keras.layers import Dense, Input, Concatenate

def build_hidden_layers(hidden_dims):
    layers = []
    for dim in hidden_dims:
        layers.append(Dense(dim, activation='relu'))
    return layers

def build_skip_layers(hidden_dims):
    layers = []
    for dim in hidden_dims:
        layers.append(Dense(dim, activation='relu'))
    return layers
