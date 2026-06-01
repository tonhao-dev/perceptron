from time import sleep

def heaviside(value: any):
    if value is None: return None
    
    return 1 if value >= 0 else 0
    

def sum_vector(vector1: list[float], vector2: list[float]) -> list[float]:
    
    if len(vector1) != len(vector2):
        return []
    
    vector_result = []
    for i in range(0, len(vector1)):
        vector_result.append(vector1[i] + vector2[i])
        
    
    return vector_result
    

def multiply_vector(vector1: list[float], vector2: list[float]) -> None or float:
    
    if len(vector1) != len(vector2):
        return None
    
    result = 0
    for i in range(0, len(vector1)):
        result += vector1[i] * vector2[i]
        

    return result


def multiply_value_by_vector(value: float, vector: list[float]) -> list[float]:
    result = [] 
    
    for i in range(0, len(vector)):
        result.append(vector[i] * value)
    

    return result
    

class Neuron():
    weights: list[float]
    
    def __init__(self, weights: list[float] = []):
        self.weights = weights


class Perceptron():
    activation_function = staticmethod(heaviside)
    iteration = 1 
    
    def training(self, neuron: Neuron, inputs: list[list[float]], desired: list[float], learning_coefficient: float, bias = 1):
        input_with_bias = []
        for row in inputs:
            input_with_bias.append( row + [bias])
        
        while not self.is_converged(neuron, input_with_bias, desired):
            for i in range(len(input_with_bias)): 
                self.iteration += 1
                
                yi = self.activation_function(multiply_vector(neuron.weights, input_with_bias[i]))
                
                e = desired[i] - yi
                
                if not isinstance(yi, int):
                    continue
                
                neuron.weights = sum_vector(neuron.weights, multiply_value_by_vector(learning_coefficient * e, input_with_bias[i]))
                
                print('Pesos Atualizados = ', neuron.weights)
                
                sleep(0.5)

    def is_converged(self, neuron: Neuron, inputs: list[list[float]], desired: list[float]): 
        flag = True
        
        for i in range(0, len(inputs)):
            yi = self.activation_function(multiply_vector(neuron.weights, inputs[i]))
            
            e = yi - desired[i]
        
            if e != 0:
                flag = False
                break
            
        return flag


if __name__ == '__main__':
    and_logic_port_neuron = Neuron([0, 0, 0])
    
    inputs = [[2, 2], [1, -2], [-2, 2], [-1, 0]]
    
    desired = [0, 1, 0, 1]
    
    learning_coefficient = 1
    
    perceptron = Perceptron()
    
    print('Pesos Iniciais = ', and_logic_port_neuron.weights)
    
    perceptron.training(and_logic_port_neuron, inputs, desired, learning_coefficient)
    
    for i in range(len(and_logic_port_neuron.weights)):
        print(f'{and_logic_port_neuron.weights[i]},', end='')
        
    print()