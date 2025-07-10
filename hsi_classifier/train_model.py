# import os
# from tensorflow.keras.models import Model

# from tensorflow.keras.layers import (
#     BatchNormalization,
#     Conv2D,
#     Conv3D,
#     Dense,
#     Dropout,
#     Flatten,
#     GlobalAveragePooling2D,
#     Input,
#     LayerNormalization,
#     LeakyReLU,
#     MaxPooling2D,
#     MultiHeadAttention,
#     Reshape,
# )


# def sandwich_block(x, filters, kernel_size=(3, 3), pool_size=(2, 2), use_pooling=True, dropout_rate=0.0):
#     """Create a sandwich block with Conv2D, BatchNorm, LeakyReLU, optional MaxPooling and Dropout."""
#     x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#     if use_pooling:
#         x = MaxPooling2D(pool_size=pool_size)(x)
#     if dropout_rate > 0.0:
#         x = Dropout(dropout_rate)(x)
#     return x

# def create_model(S, L, dataset):
#     output_units = 9 if dataset in ['PU', 'PC'] else 16

#     input_layer = Input((S, S, L, 1))

#     # 3D Convolutional layers
#     x = Conv3D(filters=32, kernel_size=(3, 3, 7), activation='relu')(input_layer)
#     x = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(x)
#     x = Conv3D(filters=8, kernel_size=(3, 3, 5), activation='relu')(x)

#     # Reshape to 2D
#     x_shape = x.shape
#     x = Reshape((x_shape[1], x_shape[2], x_shape[3] * x_shape[4]))(x)

#     # First sandwich block
#     x = sandwich_block(x, filters=64, kernel_size=(3, 3), pool_size=(2, 2), use_pooling=False)

#     # Self-attention layer
#     attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
#     x = LayerNormalization(epsilon=1e-6)(attn_output + x)

#     # Second sandwich block
#     x = sandwich_block(x, filters=32, kernel_size=(3, 3), pool_size=(2, 2), use_pooling=True)

#     # Dropout and pooling
#     x = Dropout(0.4)(x)
#     x = GlobalAveragePooling2D()(x)
#     x = Flatten()(x)

#     # Dense layers
#     x = Dense(units=256, activation='relu')(x)
#     x = LeakyReLU()(x)
#     x = Dropout(0.4)(x)
#     x = Dense(units=128, activation='relu')(x)
#     x = LeakyReLU()(x)

#     # Output layer
#     output_layer = Dense(units=output_units, activation='softmax')(x)

#     return Model(inputs=input_layer, outputs=output_layer)

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# def train_model(model, save_folder, dataset, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
#     """
#     Trains a compiled Keras model with checkpointing and optional early stopping.

#     Args:
#         model: A compiled Keras model.
#         dataset (str): Dataset name for naming the saved model.
#         X_train, y_train: Training data and labels.
#         X_val, y_val: Optional validation data and labels.
#         epochs (int): Number of training epochs.
#         batch_size (int): Batch size for training.

#     Returns:
#         model: Trained Keras model.
#         history: Training history object.
#     """
#     # Define callbacks
#     monitor_metric = 'val_acc' if X_val is not None else 'acc'
#     os.makedirs(os.path.dirname(save_folder), exist_ok=True)
#     file_name = f"best-model-{dataset}.keras"
#     filepath = os.path.join(save_folder, file_name)

#     # filepath = f"./data/04_models/best-model-{dataset}.keras"

#     checkpoint = ModelCheckpoint(
#         filepath=filepath,
#         monitor=monitor_metric,
#         verbose=1,
#         save_best_only=True,
#         mode='max'
#     )

#     early_stop = EarlyStopping(
#         monitor='val_loss',
#         patience=10,
#         restore_best_weights=True
#     )

#     callbacks_list = [checkpoint, early_stop]

#     # Train the model
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val) if X_val is not None else None,
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=callbacks_list,
#         verbose=1
#     )

#     return model, history
