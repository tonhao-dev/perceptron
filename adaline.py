import csv
import random
import matplotlib.pyplot as plt


random.seed("Luís Antônio - PPGCC 2026")


def sum_vector(vector1: list[float], vector2: list[float]) -> list[float]:
    if len(vector1) != len(vector2):
        raise ValueError("Os vetores devem ter o mesmo tamanho")

    vector_result = []
    for i in range(0, len(vector1)):
        vector_result.append(vector1[i] + vector2[i])

    return vector_result


def multiply_vector(vector1: list[float], vector2: list[float]) -> float:
    if len(vector1) != len(vector2):
        raise ValueError("Os vetores devem ter o mesmo tamanho")

    result = 0
    for i in range(0, len(vector1)):
        result += vector1[i] * vector2[i]

    return result


def multiply_value_by_vector(value: float, vector: list[float]) -> list[float]:
    result = []

    for i in range(0, len(vector)):
        result.append(vector[i] * value)

    return result


class Neuron:
    weights: list[float]

    def __init__(self, weights: list[float] | None = None):
        self.weights = weights if weights is not None else []


class Adaline:
    def __init__(self):
        self.iteration: int = 1
        self.mse_history: list[float] = []

    def train(
        self,
        neuron: Neuron,
        inputs: list[list[float]],
        desired: list[float],
        learning_coefficient: float,
        bias=1,
        max_epochs=300,
    ):
        input_with_bias = []
        for row in inputs:
            input_with_bias.append(row + [bias])

        initial_mse = self.calculate_mse(neuron, input_with_bias, desired)
        self.mse_history.append(initial_mse)

        for epoch in range(0, max_epochs):
            for i in range(len(input_with_bias)):
                self.iteration += 1

                yi = multiply_vector(neuron.weights, input_with_bias[i])

                e = desired[i] - yi

                neuron.weights = sum_vector(
                    neuron.weights,
                    multiply_value_by_vector(
                        learning_coefficient * e, input_with_bias[i]
                    ),
                )

            mse = self.calculate_mse(neuron, input_with_bias, desired)
            self.mse_history.append(mse)

    def calculate_mse(
        self, neuron: Neuron, inputs: list[list[float]], desired: list[float]
    ) -> float:
        total_error = 0.0

        for i in range(len(inputs)):
            yi = multiply_vector(neuron.weights, inputs[i])
            e = desired[i] - yi
            total_error += e**2

        return total_error / len(inputs)

    def check(
        self, neuron: Neuron, inputs: list[list[float]], desired: list[float], bias=1
    ) -> float:
        hits = 0

        for i in range(0, len(inputs)):
            result = multiply_vector(neuron.weights, inputs[i] + [bias])

            if result >= 0 and desired[i] == 1:
                hits += 1
            elif result < 0 and desired[i] == -1:
                hits += 1

        return hits / len(inputs)


adaline = Adaline()

# We will consider just sepal_length and petal_length
setosa_classification_neuron = Neuron([0, 0, 0])

learning_coefficient = 0.001

bias = 1

max_epochs = 30

with open("iris.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Just to jump the header line

    dataset = [[float(row[1]), float(row[3])] + [row[5]] for row in reader]

    # 40 -> Iris-setosa, 40 -> Iris-versicolor, 40 -> Iris-virginica
    eighty_percent_dataset = dataset[0:40] + dataset[50:90] + dataset[100:140]
    random.shuffle(eighty_percent_dataset)

    # 10 -> Iris-setosa, 10 -> Iris-versicolor, 10 -> Iris-virginica
    twenty_percent_dataset = dataset[40:50] + dataset[90:100] + dataset[140:150]
    random.shuffle(twenty_percent_dataset)

    training_inputs = [[row[0], row[1]] for row in eighty_percent_dataset]
    test_inputs = [[row[0], row[1]] for row in twenty_percent_dataset]

    desired = [1 if row[2] == "Iris-setosa" else -1 for row in eighty_percent_dataset]
    desired_test = [
        1 if row[2] == "Iris-setosa" else -1 for row in twenty_percent_dataset
    ]

    print("Pesos Iniciais = ", setosa_classification_neuron.weights)

    adaline.train(
        setosa_classification_neuron,
        training_inputs,
        desired,
        learning_coefficient,
        bias,
        max_epochs,
    )

    print("Pesos Após Treino = ", setosa_classification_neuron.weights)

    training_accuracy = adaline.check(
        setosa_classification_neuron,
        training_inputs,
        desired,
    )

    test_accuracy = adaline.check(
        setosa_classification_neuron,
        test_inputs,
        desired_test,
    )

    print(f"Acurácia no treino: {training_accuracy * 100:.2f}%")
    print(f"Acurácia no teste: {test_accuracy * 100:.2f}%")
    print(f"MSE inicial: {adaline.mse_history[0]}")
    print(f"MSE final: {adaline.mse_history[-1]}")
    print(f"Épocas executadas: {len(adaline.mse_history) - 1}")

plt.figure(figsize=(8, 5))

plt.plot(range(len(adaline.mse_history)), adaline.mse_history, marker="o")

plt.title("Gráfico de evolução do MSE")
plt.xlabel("Época")
plt.ylabel("Mean Squared Error")
plt.grid(True)

plt.show()
