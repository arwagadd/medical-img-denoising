from matplotlib import pyplot as plt
import tensorflow as tf
from math import prod
import numpy as np
import os


def init_data(path: str) -> tuple[np.ndarray]:
    """ """
    ### load the X ###
    X_path: str = os.path.join(path, "X.npy")
    X: np.ndarray = np.load(X_path)

    ### load the y ###
    y_path: str = os.path.join(path, "y.npy")
    y: np.ndarray = np.load(y_path)

    return X, y


def init_model(image_shape: tuple[int]) -> tf.keras.Model:
    input_shape = image_shape

    latent_dim = 100

    encoder = create_encoder_vae(input_shape, latent_dim)

    decoder = create_decoder_vae(latent_dim, output_shape=image_shape)

    # Input layer for decoder
    decoder_input = tf.keras.Input(shape=(latent_dim,))

    # Connect encoder output to decoder input
    decoder_output = decoder(encoder(encoder.input)[0])

    # Define VAE model
    vae = tf.keras.Model(encoder.input, decoder_output, name="vae")

    return vae


def create_encoder_vae(input_shape, latent_dim):
    encoder_input = tf.keras.Input(shape=input_shape)
    vgg_layers = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_tensor=encoder_input
    )
    # Freeze the first 10 layers of VGG
    for layer in vgg_layers.layers[:10]:
        layer.trainable = False
    # Extract output from VGG
    vgg_output = vgg_layers.output
    # Additional layers to further process the features extracted by VGG16
    x = tf.keras.layers.Flatten()(vgg_output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    # Define VAE layers (mean and logvar)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_logvar = tf.keras.layers.Dense(latent_dim)(x)
    # Define the encoder model
    encoder = tf.keras.Model(encoder_input, [z_mean, z_logvar], name="encoder")
    return encoder


def create_decoder_vae(latent_dim, output_shape):
    # Input layer (latent space representation)
    decoder_input = tf.keras.Input(shape=(latent_dim,))

    # Dense layer to map the latent space representation to the required size
    x = tf.keras.layers.Dense(512, activation="relu")(decoder_input)
    x = tf.keras.layers.Dense(prod(output_shape), activation="relu")(
        x
    )  # Adjust the size according to your output shape

    # Reshape the output to match the required output
    x = tf.keras.layers.Reshape((300, 300, 3))(x)

    # Convolutional layers to reconstruct the image
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(
        x
    )
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(
        x
    )
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(
        x
    )

    # Output layer with the desired output shape
    decoder_output = tf.keras.layers.Conv2DTranspose(
        3, (3, 3), activation="sigmoid", padding="same"
    )(x)

    # Define the decoder model
    decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")
    return decoder


def main() -> int:
    """ """
    ### init data ###
    path: str = "./dbs/intermittent/"
    X, y = init_data(path)
    batch_size: int = 16
    epochs: int = 10

    ### get the image shape ###
    image_shape: tuple[int] = X.shape[1:]

    ### init model ###
    vae: tf.keras.Model = init_model(image_shape)

    ### compile the model ###
    loss: tf.keras.losses.Loss = tf.keras.losses.KLDivergence(
        reduction="sum_over_batch_size", name="kl_divergence"
    )
    vae.compile(optimizer="adam", loss=loss)

    ### fit the model ###
    history: tf.keras.callbacks.History = vae.fit(X, y, batch_size=batch_size, epochs=epochs)

    ### show the history ###
    plt.plot(history.history["loss"])
    plt.show()

    ### predict ###
    breakpoint()
    x: np.ndarray = np.random.choice(X, size=1)
    y_pred: np.ndarray = vae.predict(x)

    ### show the prediction ###
    plt.subplot(1, 2, 1)
    plt.imshow(x[0])






    return 0


if __name__ == "__main__":
    main()
#   __   _,_ /_ __,
# _(_/__(_/_/_)(_/(_
#  _/_
# (/
