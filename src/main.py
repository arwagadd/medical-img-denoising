import numpy as np
import tensorflow as tf
# from dSRVAE.modules import VAE_denoise
import os


def init_data(path: str, batch_size: int) -> tuple[np.ndarray]:
    x_data = []
    y_data = []
    for file_name in os.listdir(path):
        if file_name.startswith("x_") and file_name.endswith(".npy"):
            x_batch = np.load(os.path.join(path, file_name))
            if x_batch.shape[0] != batch_size:
                raise ValueError(f"Unexpected batch size {x_batch.shape[0]} for file {file_name}")
            x_data.append(x_batch)
        elif file_name.startswith("y_") and file_name.endswith(".npy"):
            y_batch = np.load(os.path.join(path, file_name))
            if y_batch.shape[0] != batch_size:
                raise ValueError(f"Unexpected batch size {y_batch.shape[0]} for file {file_name}")
            y_data.append(y_batch)
    return np.array(x_data), np.array(y_data)


def init_model(image_shape : tuple[int]) -> tf.keras.Model:

    input_shape = image_shape
    
    latent_dim = 100  

    encoder = create_encoder_vae(input_shape, latent_dim)

    decoder = create_decoder_vae(latent_dim, output_shape=image_shape)

    # Input layer for decoder
    decoder_input = tf.keras.Input(shape=(latent_dim,))
    
    # Connect encoder output to decoder input
    decoder_output = decoder(encoder(encoder.input)[0])

    # Define VAE model
    vae = tf.keras.Model(encoder.input, decoder_output, name='vae')
    
    return vae

def create_encoder_vae(input_shape, latent_dim):
    
    encoder_input = tf.keras.Input(shape=input_shape)  
    vgg_layers = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=encoder_input)
    # Freeze the first 10 layers of VGG
    for layer in vgg_layers.layers[:10]:  
        layer.trainable = False
    # Extract output from VGG
    vgg_output = vgg_layers.output
    # Additional layers to further process the features extracted by VGG16
    x = tf.keras.layers.Flatten()(vgg_output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    # Define VAE layers (mean and logvar)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_logvar = tf.keras.layers.Dense(latent_dim)(x)
    # Define the encoder model
    encoder = tf.keras.Model(encoder_input, [z_mean, z_logvar], name='encoder')
    return encoder

def create_decoder_vae(latent_dim, output_shape):
    # Input layer (latent space representation)
    decoder_input = tf.keras.Input(shape=(latent_dim,))
    
    # Dense layer to map the latent space representation to the required size
    x = tf.keras.layers.Dense(512, activation='relu')(decoder_input)
    x = tf.keras.layers.Dense(300 * 300 * 3, activation='relu')(x)  # Adjust the size according to your output shape
    
    # Reshape the output to match the required output
    x = tf.keras.layers.Reshape((300, 300, 3))(x)
    
    # Convolutional layers to reconstruct the image
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    
    # Output layer with the desired output shape
    decoder_output = tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Define the decoder model
    decoder = tf.keras.Model(decoder_input, decoder_output, name='decoder')
    return decoder

def main():
    # data = init_data("dbs/intermittent", batch_size=9)

    # Encoder
    input_shape = (300, 300, 3)
    latent_dim = 100  # Define the latent dimension
    encoder = create_encoder_vae(input_shape, latent_dim)
    encoder.summary()

    # Decoder
    output_shape = (300, 300, 3)  # Define the output shape of the decoder
    decoder = create_decoder_vae(latent_dim, output_shape)
    decoder.summary()

    # Complete model
    vae_model = init_model(image_shape=(300, 300, 3))
    vae_model.summary()
if __name__ == "__main__":
    main()
