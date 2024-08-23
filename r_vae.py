# define r_vae

import tensorflow as tf
from enc_dec import Encoder, Decoder
from rep_trick import reparameterize

class RVAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, hidden_dims_enc, hidden_dims_dec):
        super(RVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims_enc, latent_dim).build()
        self.decoder = Decoder(latent_dim, hidden_dims_dec, input_dim).build()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

    def compute_loss(self, x, y, y_pred, z_mean, z_log_var):
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(x - y_pred))
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        # Conditional KL divergence loss
        conditional_kl_loss = tf.reduce_mean(tf.square(z_mean - y))
        total_loss = reconstruction_loss + kl_loss + conditional_kl_loss
        return total_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, z_mean, z_log_var = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, z_mean, z_log_var)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}
    
    def test_step(self, data):
        x, y = data
        y_pred, z_mean, z_log_var = self(x, training=False)
        loss = self.compute_loss(x, y, y_pred, z_mean, z_log_var)
        return {'loss': loss}
