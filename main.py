from time import sleep

def heaviside(value: float):
    return 1 if value >= 0 else 0
    

def sum_vector(vector1: list[float], vector2: list[float]) -> list[float]:
    print(f'sum_vector -> vector1 -> {vector1}')
    print(f'sum_vector -> vector2 -> {vector2}')
    
    if len(vector1) != len(vector2):
        return []
    
    vector_result = []
    for i in range(0, len(vector1)):
        vector_result.append(vector1[i] + vector2[i])
        
    print(f'sum_vector -> vector_result -> {vector_result}')    
    
    return vector_result
    

def multiply_vector(vector1: list[float], vector2: list[float]) -> None or float:
    print(f'multiply_vector -> vector1 -> {vector1}')
    print(f'multiply_vector -> vector2 -> {vector2}')
    
    if len(vector1) != len(vector2):
        return None
    
    result = 0
    for i in range(0, len(vector1)):
        result += vector1[i] * vector2[i]
        
    print(f'multiply_vector -> result -> {result}')

    return result
    

def multiply_matrix(matrix1: list[list[float]], matrix2: list[list[float]]):    
    if(len(matrix1) != len(matrix2[0])):
        prfloat('A quantidade de colunas da 1° matriz deve ser igual a quantidade de linhas da 2°')
        return None
    
    accumulator = 0
    for i in range(0, len(matrix1)):
        accumulator = matrix1[0][i]  * matrix2[i][0]


def multiply_value_by_vector(value: float, vector: list[float]) -> list[float]:
    result = 0 
    
    for i in range(0, len(vector)):
        vector[i] = vector[i] * value
    
    print(f'multiply_value_by_vector -> vector -> {vector}')

    return vector
    

class Neuron():
    weights: list[float]
    
    def __init__(self, weights: list[float] = []):
        self.weights = weights


class Perceptron():
    activation_function = staticmethod(heaviside)
    iteration = 1 
    
    def training(self, neuron: Neuron, inputs: list[list[float]], targets: list[float], learning_coefficient: float):
        while not self.is_converged(neuron, inputs, targets):
            for i in range(len(inputs)): 
                print(f'------------- {self.iteration}° iteração -----------')
                print(f'pesos = [{neuron.weights}]')
                self.iteration += 1
                
                yi = self.activation_function(multiply_vector(neuron.weights, inputs[i]))
                
                e = targets[i] - yi
                
                print(f'is_converged -> yi -> {yi}')
                print(f'is_converged -> targets[i] -> {targets[i]}')
                print(f'is_converged -> e -> {e}')
                
                if not isinstance(yi, int):
                    print(f'{yi} não é float!')
                    continue
                
                neuron.weights = sum_vector(neuron.weights, multiply_value_by_vector(learning_coefficient * e, inputs[i]))
                
                sleep(1)
                print('---------------------------------')

    def is_converged(self, neuron: Neuron, inputs: list[list[float]], targets: list[float]): 
        flag = True
        
        for i in range(0, len(inputs)):
            yi = self.activation_function(multiply_vector(neuron.weights, inputs[i]))
            
            e = yi - targets[i]
        
            if e != 0:
                flag = False
                break
            
        return flag


if __name__ == '__main__':
    and_logic_port_neuron = Neuron([-0.1, -0.2, 0.2])
    
    inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    
    targets = [0, 0, 0, 1]
    
    learning_coefficient = 0.5
    
    perceptron = Perceptron()
    
    perceptron.training(and_logic_port_neuron, inputs, targets, learning_coefficient)
    
    for i in range(len(and_logic_port_neuron.weights)):
        print(f'{and_logic_port_neuron.weights[i]},', end='')
        
    print()