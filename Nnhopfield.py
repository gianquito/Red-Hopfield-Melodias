import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class HopfieldNetwork:
    """
    Implementación de una Red de Hopfield para el reconocimiento de patrones musicales.
    La red puede aprender patrones binarios y recuperarlos incluso cuando están distorsionados con ruido.

    Attributes:
        size (int): Tamaño de los patrones de entrada (número de neuronas en la red)
        weights (numpy.ndarray): Matriz de pesos sinápticos de la red
    """

    def __init__(self, size):
        """
        Inicializa la red con un tamaño específico.

        Args:
            size (int): Número de neuronas en la red
        """
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """
        Entrena la red con un conjunto de patrones usando la regla de Hebbian.

        Args:
            patterns (numpy.ndarray): Matriz donde cada fila es un patrón de entrenamiento
        """
        for pattern in patterns:
            # Calcula el producto exterior para cada patrón
            self.weights += np.outer(pattern, pattern)
        # Elimina las autoconexiones
        np.fill_diagonal(self.weights, 0)
        # Normaliza los pesos por el número de patrones
        self.weights /= len(patterns)

    def recall(self, pattern, max_iter=10):
        """
        Recupera un patrón almacenado a partir de un patrón de entrada.

        Args:
            pattern (numpy.ndarray): Patrón de entrada a recuperar
            max_iter (int): Número máximo de iteraciones para la convergencia

        Returns:
            numpy.ndarray: Patrón recuperado
        """
        pattern = pattern.copy()
        for _ in range(max_iter):
            previous_pattern = pattern.copy()
            for i in range(self.size):
                # Actualiza cada neurona según la regla de activación
                pattern[i] = 1 if np.dot(self.weights[i], pattern) >= 0 else -1
            # Verifica si el patrón se ha estabilizado
            if np.array_equal(pattern, previous_pattern):
                break
        return pattern

    def add_noise(self, pattern, noise_level=0.3):
        """
        Añade ruido aleatorio a un patrón.

        Args:
            pattern (numpy.ndarray): Patrón original
            noise_level (float): Proporción de bits que serán invertidos (0-1)

        Returns:
            numpy.ndarray: Patrón con ruido
        """
        noisy_pattern = pattern.copy()
        num_noisy_bits = int(self.size * noise_level)
        noise_indices = np.random.choice(range(self.size), num_noisy_bits, replace=False)
        noisy_pattern[noise_indices] *= -1
        return noisy_pattern

def plot_patterns(original, noisy, recovered, title=""):
    """
    Visualiza los patrones original, con ruido y recuperado.

    Args:
        original (numpy.ndarray): Patrón original
        noisy (numpy.ndarray): Patrón con ruido
        recovered (numpy.ndarray): Patrón recuperado
        title (str): Título para la visualización
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original.reshape(8, 8), cmap="bwr")
    ax[0].set_title("Melodía Original")
    ax[1].imshow(noisy.reshape(8, 8), cmap="bwr")
    ax[1].set_title("Melodía con Ruido")
    ax[2].imshow(recovered.reshape(8, 8), cmap="bwr")
    ax[2].set_title("Melodía Recuperado")
    plt.suptitle(title)
    plt.show()

def test_melody_recognition(network, pattern, melody_name, noise_level=0.3):
    """
    Prueba la capacidad de la red para reconocer una melodía con ruido.

    Args:
        network (HopfieldNetwork): Red de Hopfield entrenada
        pattern (numpy.ndarray): Patrón de melodía original
        melody_name (str): Nombre de la melodía para la visualización
        noise_level (float): Nivel de ruido a aplicar

    Returns:
        numpy.ndarray: Patrón recuperado
    """
    noisy_pattern = network.add_noise(pattern, noise_level)
    recovered_pattern = network.recall(noisy_pattern)
    plot_patterns(pattern, noisy_pattern, recovered_pattern,
                 title=f"Reconstrucción de {melody_name}")
    return recovered_pattern

def compare_patterns(recovered, original, all_patterns):
    """
    Compara el patrón recuperado con el original y todos los patrones conocidos.

    Args:
        recovered (numpy.ndarray): Patrón recuperado
        original (numpy.ndarray): Patrón original
        all_patterns (numpy.ndarray): Todos los patrones de entrenamiento
    """
    if np.array_equal(recovered, original):
        print("La melodía recuperada coincide con la melodía original")
    else:
        similar_found = False
        for idx, pattern in enumerate(all_patterns):
            if np.array_equal(recovered, pattern):
                print(f"La melodía recuperada coincide con otra melodía en el dataset (índice {idx})")
                similar_found = True
                break
        if not similar_found:
            print("La melodía recuperada no coincide completamente con ninguna de las melodías conocidas")


# Definimos los argumentos para ejecutar el programa
parser = argparse.ArgumentParser(
                    prog='Red de Hopfield')
parser.add_argument('datasetPath', help="La ruta al dataset")
parser.add_argument('--noise1', type=float, default=0.4, help="El nivel de ruido a aplicarle al primer patrón. Es un valor entre 0 y 1")  
parser.add_argument('--noise2', type=float, default=0.3, help="El nivel de ruido a aplicarle al segundo patrón. Es un valor entre 0 y 1")  

args = parser.parse_args()

# Carga y preprocesamiento de datos
data = pd.read_csv(args.datasetPath, header=None)
data = data.iloc[1:, 1:]  # Elimina la primera fila (cabecera) y la primera columna
patterns = data.values.astype(int)  # Convierte el DataFrame a una matriz de numpy

# Inicialización y entrenamiento de la red
hopfield_net = HopfieldNetwork(size=patterns.shape[1])
hopfield_net.train(patterns)

# Ejemplo de uso
print("Probando recuperación de melodías con ruido utilizando una red de hopfield")

# Prueba con el primer patrón
recovered_pattern_1 = test_melody_recognition(hopfield_net, patterns[0],
                                             "Primera melodía", noise_level=args.noise1)
compare_patterns(recovered_pattern_1, patterns[0], patterns)

# Prueba con el segundo patrón
recovered_pattern_2 = test_melody_recognition(hopfield_net, patterns[25],
                                             "Segunda melodía", noise_level=args.noise2)
compare_patterns(recovered_pattern_2, patterns[25], patterns)