import csv
import numpy as np


def preprocess_data(file_path):
    texts = []
    languages = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            text = row[1].lower()
            texts.append(text)
            languages.append(row[0])
    return texts, languages


def count_occurrences(text):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    counts = np.zeros(26)
    for char in text:
        if char in alphabet:
            index = alphabet.index(char)
            counts[index] += 1
    return counts


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector


def generate_input_vectors(texts):
    input_vectors = np.array([normalize_vector(count_occurrences(text)) for text in texts])
    return input_vectors


class SingleLayerNetwork:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)

    def train(self, X, y, learning_rate=0.01, epochs=100):
        one_hot_y = np.eye(self.weights.shape[1])[y]
        for _ in range(epochs):
            predictions = self.predict(X)
            error = one_hot_y - predictions
            self.weights += learning_rate * np.dot(X.T, error)

    def predict(self, X):
        return np.dot(X, self.weights)


def calc_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    correct = np.sum(np.argmax(predictions, axis=1) == y_test)
    total = len(y_test)
    accuracy = (correct / total) * 100
    return accuracy


def classify_text(text, model):
    input_vector = generate_input_vectors([text])
    prediction = model.predict(input_vector)
    language_index = np.argmax(prediction)
    return language_index


if __name__ == "__main__":
    train_texts, train_languages = preprocess_data('lang.train.csv')
    test_texts, test_languages = preprocess_data('lang.test.csv')

    X_train = generate_input_vectors(train_texts)
    X_test = generate_input_vectors(test_texts)

    unique_languages = set(train_languages)
    language_map = {language: idx for idx, language in enumerate(unique_languages)}
    y_train = np.array([language_map[language] for language in train_languages])
    y_test = np.array([language_map[language] for language in test_languages])

    input_size = 26
    output_size = len(language_map)
    model = SingleLayerNetwork(input_size, output_size)
    model.train(X_train, y_train)

    while True:
        print("\nMenu:")
        print("1. Show accuracy")
        print("2. Show misclassified texts")
        print("3. Try your own text")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            test_accuracy = calc_accuracy(model, X_test, y_test)
            print("Test accuracy:", test_accuracy)
        elif choice == '2':
            predictions = model.predict(X_test)
            misclassified_texts = []
            for i in range(len(test_texts)):
                predicted_language = np.argmax(predictions[i])
                true_language = y_test[i]
                if predicted_language != true_language:
                    misclassified_texts.append(
                        (test_texts[i], list(language_map.keys())[true_language],
                         list(language_map.keys())[predicted_language]))
            print("\nMisclassified Texts:")
            for text, true_language, predicted_language in misclassified_texts:
                print("Text:", text)
                print("True Language:", true_language)
                print("Predicted Language:", predicted_language)
                print()
        elif choice == '3':
            text = input("Enter a text to classify (type 'exit' to quit): ")
            if text.lower() == 'exit':
                break
            language_index = classify_text(text, model)
            languages = list(language_map.keys())
            print("Predicted language:", languages[language_index])
        elif choice == '4':
            print("Bye-bye...")
            break
        else:
            print("Invalid choice. Please, try again.")