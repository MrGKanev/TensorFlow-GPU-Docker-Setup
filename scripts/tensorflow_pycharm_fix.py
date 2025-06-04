"""
TensorFlow PyCharm Integration Fix

This script demonstrates how to fix the common issue with TensorFlow
optimizers and LSTM models in TensorFlow 2.11.0.

The key issues addressed are:
1. Using legacy optimizers instead of experimental ones
2. Fixing shape mismatches in LSTM input
3. Setting up proper GPU memory management
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy optimizer

# Fix 1: Configure GPU memory growth
def configure_gpu():
    """Configure GPU to use memory growth and prevent OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled on {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    
    # Check if GPU is available after configuration
    gpus_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"GPU available: {gpus_available}")
    return gpus_available

# Fix 2: Create GAN with proper optimizer configuration
def create_gan_model(latent_dim=100):
    """Create a GAN with properly configured optimizers."""
    # Generator
    generator = models.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(0.2),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dense(1024),
        layers.LeakyReLU(0.2),
        layers.Dense(28*28, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    
    # Discriminator
    discriminator = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(1024),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Dense(256),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Fix 3: Use legacy optimizer instead of experimental
    discriminator.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),  # Use legacy Adam
        loss='binary_crossentropy'
    )
    
    # GAN
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output)
    
    # Fix 3: Use legacy optimizer for the GAN too
    gan.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),  # Use legacy Adam
        loss='binary_crossentropy'
    )
    
    return generator, discriminator, gan

# Fix 4: For LSTM models specifically
def create_lstm_model(sequence_length=30, features=1):
    """Create an LSTM model with proper shape handling."""
    model = models.Sequential([
        # Fix input shape specification
        layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
        layers.LSTM(50),
        layers.Dense(1)
    ])
    
    # Fix 3: Use legacy optimizer
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    
    return model

# Example training function with proper shape handling
def train_gan(generator, discriminator, gan, epochs=10, batch_size=128):
    """Train a GAN with proper batch handling."""
    # Generate fake data for demonstration
    # In a real scenario, you'd load your dataset here
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Generate random noise
        noise = np.random.normal(0, 1, (batch_size, 100))
        
        # Generate fake images
        generated_images = generator.predict(noise)
        
        # Fix 5: Ensure proper shape
        # For GAN training with image data
        if len(generated_images.shape) == 3:
            # Add channel dimension if missing
            generated_images = np.expand_dims(generated_images, axis=-1)
            
        # Create labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(
            # Use random data for demo - in real code use your dataset
            np.random.random((batch_size, 28, 28, 1)), 
            real_labels
        )
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real_labels)
        
        print(f"  D loss: {d_loss}, G loss: {g_loss}")
        
# Example: Fix LSTM model training
def train_lstm_model(model, epochs=10, batch_size=32):
    """Train an LSTM model with proper shape handling."""
    # Generate random time series data for demonstration
    sequence_length = 30
    n_samples = 1000
    
    # Create sample data
    X = np.random.random((n_samples, sequence_length, 1))
    y = np.random.random((n_samples, 1))
    
    # Fix 6: Make sure input shape is consistent
    print(f"Model expects input shape: {model.input_shape}")
    print(f"Actual input shape: {X.shape}")
    
    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Example of how to run the code
if __name__ == "__main__":
    # Step 1: Configure GPU
    configure_gpu()
    
    # Example 1: Fix GAN training
    print("\nExample 1: Creating and training GAN")
    generator, discriminator, gan = create_gan_model()
    # Uncomment to train (commented out to avoid long execution)
    # train_gan(generator, discriminator, gan, epochs=1)
    
    # Example 2: Fix LSTM training
    print("\nExample 2: Creating and training LSTM")
    lstm_model = create_lstm_model()
    # Uncomment to train (commented out to avoid long execution)
    # train_lstm_model(lstm_model, epochs=1)
    
    print("\nAll models created successfully with proper configuration.")
    print("✅ To fix your code, make these changes:")
    print("1. Use legacy optimizers: from tensorflow.keras.optimizers.legacy import Adam")
    print("2. Configure GPU memory growth at the start of your script")
    print("3. Ensure input shapes match the model's expected input shape")
    print("4. For LSTM models, pay special attention to sequence length in your input data")
