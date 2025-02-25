# Topic 9: Neural Networks
# Objective: Build a mini brain for complex patterns.
# What You’ll Learn: Basics of layers and activation.
# Real-World Use: Image recognition, voice assistants, or memes.
# Tip: Start simple—too many layers = chaos!
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
print(model.summary())