import csv
import math


def read_file(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            data.append(line)
    return data


def euclidean_distance(vec1, vec2):
    distance = 0
    for i in range(len(vec1) - 1):
        distance += (float(vec1[i]) - float(vec2[i])) ** 2
    return math.sqrt(distance)


def get_knn(train_set, test_instance, k):
    distances = []
    for train_instance in train_set:
        dist = euclidean_distance(train_instance, test_instance)
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])
    knn = []
    for i in range(k):
        knn.append(distances[i][0])
    return knn


def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def classify(neighbors):
    class_votes = {}
    for neighbor in neighbors:
        response = neighbor[-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]


def calculate_accuracy_and_write_to_csv(train_filename, test_filename, output_filename):
    with open(output_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        csv_writer.writerow(['k', 'Accuracy'])
        train_set = read_file(train_filename)
        test_set = read_file(test_filename)
        for k in range(1, 101):
            predictions = []
            for test_instance in test_set:
                neighbors = get_knn(train_set, test_instance, k)
                result = classify(neighbors)
                predictions.append(result)
            accuracy = get_accuracy(test_set, predictions)
            csv_writer.writerow([k, accuracy])


def main(train_filename, test_filename):
    while True:
        print("1. Show accuracy for a given file")
        print("2. Enter your a vector for classification")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            k = int(input("Enter the value of k: "))
            train_set = read_file(train_filename)
            test_set = read_file(test_filename)
            predictions = []
            for test_instance in test_set:
                neighbors = get_knn(train_set, test_instance, k)
                result = classify(neighbors)
                predictions.append(result)
            accuracy = get_accuracy(test_set, predictions)
            print("Calculated accuracy:", accuracy)

        elif choice == "2":
            train_set = read_file(train_filename)
            input_vector = input("Enter a vector (comma-separated values): ").strip().split(',')
            if len(input_vector) != len(train_set[0]) - 1:
                print("Invalid input")
            else:
                k = int(input("Enter the value of k: "))
                neighbors = get_knn(train_set, input_vector, k)
                result = classify(neighbors)
                print("Predicted class:", result)

        elif choice == "3":
            print("Bye-bye...")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main("data/iris.data", "data/iris.test.data")