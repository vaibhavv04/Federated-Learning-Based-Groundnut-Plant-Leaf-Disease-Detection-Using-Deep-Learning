import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

# Load each client's model
# client1_model = tf.keras.models.load_model('model_client_1_3x_trainable_true.keras', custom_objects={'KerasLayer': hub.KerasLayer})
model = tf.keras.Sequential([
    tf.keras.models.load_model('model_client_1_3x_trainable_true.keras', custom_objects={'KerasLayer': hub.KerasLayer})  # Specify input shape
])
client2_model = tf.keras.models.load_model('model_client_2_3x_trainable_true.keras')
client3_model = tf.keras.models.load_model('model_client_3_3x_trainable_true.keras')

# Extract weights from each model
client1_weights = client1_model.get_weights()
client2_weights = client2_model.get_weights()
client3_weights = client3_model.get_weights()

# Average the weights
global_weights = []
for weights_client1, weights_client2, weights_client3 in zip(client1_weights, client2_weights, client3_weights):
    avg_weights = np.mean([weights_client1, weights_client2, weights_client3], axis=0)
    global_weights.append(avg_weights)

# Create a new global model with the same architecture as the clients
global_model = tf.keras.models.clone_model(client1_model)
global_model.set_weights(global_weights)

# Save the global model to a file
global_model.save('global_model.keras')
