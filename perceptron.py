import numpy as np


class Perceptron:

    def __init__(self, input_size, lr=1.0):
        # Cria o vetor de pesos preenchido com zeros.
        # Adicionamos '+ 1' no tamanho para incluir o
        # peso do Bias (w0) na primeira posição.
        self.weights = np.zeros(input_size + 1)
        # Define a taxa de aprendizagem (passo do ajuste dos pesos)
        self.lr = lr

    def activation_function(self, v):
        # Função Limiar (Hard Limiter): se a soma (v) for
        # maior ou igual a zero o neurônio dispara '1' (ativado).
        # Se for menor que zero, dispara '0'.
        return 1 if v >= 0 else 0

    def predict(self, x):
        # Injeta o valor fixo do Bias (+1.0) no início
        # do vetor de características
        x_with_bias = np.insert(x, 0, 1.0)
        # Realiza o produto escalar: multiplica cada entrada
        # pelo seu peso correspondente e soma tudo
        v = np.dot(self.weights, x_with_bias)
        # Passa o resultado da soma pela função limiar para
        # obter a resposta final (0 ou 1)
        return self.activation_function(v)

    def train(self, X, d, max_epochs=100):
        # Converte as listas de treino em estruturas do NumPy
        # para permitir cálculos matemáticos diretos
        X = np.array(X)
        d = np.array(d)

        for epoch in range(max_epochs):
            errors = 0
            print(f"Época {epoch + 1} \n - Pesos Iniciais: {self.weights}")

            # Varre cada linha (amostra) da tabela de dados sequencialmente
            for i in range(len(X)):
                x_i = X[i]
                x_with_bias = np.insert(x_i, 0, 1.0)

                # O neurônio tenta adivinhar a classe com base nos pesos atuais
                y = self.predict(x_i)
                # Calcula a diferença entre o que o exercício
                # queria (d) e o que o neurônio previu (y)
                error = d[i] - y

                # Se houver erro (diferente de 0),
                # os pesos precisam ser ajustados
                if error != 0:
                    errors += 1
                    # Regra do Perceptron:
                    # Novo Peso = Peso Atual + (Taxa * Erro * Entrada)
                    self.weights += self.lr * error * x_with_bias

                print(
                    f"  Amostra {x_i} | Desejado (d): {d[i]} | Predito (y): {y} -> Pesos após análise: {self.weights}"
                )

            # Se ao final de uma época inteira nenhuma
            # amostra gerou erro, o algoritmo convergiu
            if errors == 0:
                print(
                    f"\n[Sucesso] Convergência atingida com sucesso na época {epoch + 1}!"
                )
                break


# Entradas baseadas nas colunas x1 e x2 da tabela
X_train = [
    [2, 2],
    [1, -2],
    [-2, 2],
    [-1, 0]
    ]

# Saídas desejadas baseadas na coluna d da tabela
d_train = [0, 1, 0, 1]

# Inicializa o classificador informando que as amostras possuem 2 entradas (x1 e x2)
perceptron = Perceptron(input_size=2, lr=1.0)

# Inicia o ciclo de treinamento
perceptron.train(X_train, d_train)

# --- Exibição Estruturada do Resultado Final ---
print("\n" + "===========================================")
print("CONFIGURAÇÃO FINAL DO PERCEPTRON")
print("\n" + "===========================================")
print(f" Peso do Bias (w0 ou b) : {perceptron.weights[0]:.2f}")
print(f" Peso da Entrada x1 (w1): {perceptron.weights[1]:.2f}")
print(f" Peso da Entrada x2 (w2): {perceptron.weights[2]:.2f}")
print("\n" + "===========================================")
print(" Equação do Combinador Linear:")
print(
    f" v = ({perceptron.weights[0]:.2f})*1 + ({perceptron.weights[1]:.2f})*x1 + ({perceptron.weights[2]:.2f})*x2"
)
print("===========================================")