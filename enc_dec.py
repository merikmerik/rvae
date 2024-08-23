# Encoder Layers: Hidden layers defined by hidden_dims_enc, outputs z_mean and z_log_var.
# Decoder Layers: Hidden layers defined by hidden_dims_dec, final output layer with linear activation.

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from hidden_layers import build_hidden_layers, build_skip_layers

class Encoder:
    def __init__(self, input_dim, hidden_dims_enc, latent_dim):
        self.input_dim = input_dim
        self.hidden_dims_enc = hidden_dims_enc
        self.latent_dim = latent_dim

    def build(self):
        inputs = Input(shape=(self.input_dim,))
        x = inputs
        for layer in build_hidden_layers(self.hidden_dims_enc):
            x = layer(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        encoder = Model(inputs, [z_mean, z_log_var], name="encoder")
        return encoder

class Decoder:
    def __init__(self, latent_dim, hidden_dims_dec, output_dim):
        self.latent_dim = latent_dim
        self.hidden_dims_dec = hidden_dims_dec
        self.output_dim = output_dim

    def build(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = latent_inputs
        for layer in build_skip_layers(self.hidden_dims_dec):
            x = layer(x)
        outputs = Dense(self.output_dim, activation='linear')(x)
        decoder = Model(latent_inputs, outputs, name="decoder")
        return decoder