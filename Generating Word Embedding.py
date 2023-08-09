embeddings_index = {}
with open('glove.6B.50d.txt') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs

# Generating embeddings
embeddings_matrix = np.zeros((vocab_size1+1, embedding_dim))
for word, i in word_index1.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embeddings_matrix[i] = embedding_vector
