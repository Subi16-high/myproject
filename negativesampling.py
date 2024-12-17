import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace with your own dataset)
corpus = [
    ("bank", "financial institution"),
    ("bank", "side of a river"),
    ("book", "printed or written work"),
    ("book", "reserve a service"),
    # Add more examples with different senses
]

# Create a vocabulary from the corpus
vocab = set([word for word, _ in corpus])

# Create a mapping from words to indices
word_to_index = {word: idx for idx, word in enumerate(vocab)}

# Convert the data into indices
X = np.array([word_to_index[word] for word, _ in corpus])
y = np.array([label for label, _ in corpus])

# Convert string labels to integers
label_to_index = {label: idx for idx, label in enumerate(set(y))}
y = np.array([label_to_index[label] for label in y])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network with an embedding layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=10, input_length=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')  # Assuming one-hot encoding of labels
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy}")

# Implement adaptive negative sampling during prediction
def adaptive_negative_sampling(word, context, model, word_to_index, vocab, num_samples=5):
    # Use the trained model to predict the sense of the word in the given context
    predicted_sense = np.argmax(model.predict(np.array([word_to_index[word]])))

    # Find other senses of the word in the training data
    all_senses = set(label for label, _ in corpus if label != predicted_sense)

    # Ensure that the sample size is not larger than the population
    num_samples = min(num_samples, len(all_senses))

    # Sample negative senses based on a probabilistic model
    negative_samples = np.random.choice(list(all_senses), size=num_samples, replace=False)

    # Combine the positive sense and negative samples
    samples = [predicted_sense] + list(negative_samples)

    return samples

# Example of using adaptive negative sampling during prediction
word_to_predict = "bank"
context_to_predict = "river"
negative_samples = adaptive_negative_sampling(word_to_predict, context_to_predict, model, word_to_index, vocab)
print(f"Word: {word_to_predict}, Context: {context_to_predict}")
print(f"Predicted Sense: {np.argmax(model.predict(np.array([word_to_index[word_to_predict]])))}")
print(f"Negative Samples: {negative_samples}")
