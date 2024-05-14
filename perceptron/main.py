import random
import csv


class Perceptron:
    def __init__(self, num_features, learning_rate, num_iterations):
        self.bias = None
        self.weights = None
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def train(self, x, d):
        self.weights = [random.random() for _ in range(self.num_features)]
        self.bias = random.random()
        for _ in range(self.num_iterations):
            for i in range(len(x)):
                output = self.predict(x[i])
                for j in range(self.num_features):
                    self.weights[j] += self.learning_rate * (d[i] - output) * x[i][j]
                self.bias -= self.learning_rate * (d[i] - output)

    def predict(self, x):
        activation = sum(self.weights[j] * x_val for j, x_val in enumerate(x)) - self.bias
        return 1 if activation >= 0 else 0

    def get_accuracy(self, x, d):
        correct = 0
        for i in range(len(x)):
            prediction = self.predict(x[i])
            if prediction == d[i]:
                correct += 1
        return correct / len(x)


def load_dataset(filename):
    dataset = []
    class_names = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            features = [float(x) for x in line[:-1]]
            label = line[-1]
            if label not in class_names:
                class_names.append(label)
            label_index = class_names.index(label)
            dataset.append((features, label_index))
    x = [sample[0] for sample in dataset]
    d = [sample[1] for sample in dataset]
    num_features = len(x[0])
    return x, d, num_features, class_names


def main():
    train_file = "data/perceptron.data"
    test_file = "data/perceptron.test.data"
    x_train, d_train, num_features, class_names = load_dataset(train_file)
    x_test, d_test, _, _ = load_dataset(test_file)

    perceptron = Perceptron(num_features, learning_rate=0.01, num_iterations=100)

    while True:
        print("Options:")
        print("1. See accuracy")
        print("2. Test your own vector")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            perceptron.train(x_train, d_train)
            accuracy = perceptron.get_accuracy(x_test, d_test)
            print("Accuracy: {:.2f}%".format(accuracy * 100))
        elif choice == '2':
            input_vector = input("Enter a custom vector (comma-separated values): ").strip().split(',')
            if len(input_vector) != len(x_train[0]):
                print("The number of values should match the number of features, please, try again.")
                continue
            test_vector = [float(x) for x in input_vector]
            prediction_index = perceptron.predict(test_vector)
            predicted_class = class_names[prediction_index]
            print("Prediction for your vector:", predicted_class)
        elif choice == '3':
            print("Bye-bye...")
            break
        else:
            print("There is no such option:( Please, try again.")


if __name__ == "__main__":
    main()
