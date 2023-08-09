model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size1+1, embedding_dim,
							input_length=max_length, weights=[
								embeddings_matrix],
							trainable=False),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Conv1D(64, 5, activation='relu'),
	tf.keras.layers.MaxPooling1D(pool_size=4),
	tf.keras.layers.LSTM(64),
	tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
			optimizer='adam', metrics=['accuracy'])
model.summary()
