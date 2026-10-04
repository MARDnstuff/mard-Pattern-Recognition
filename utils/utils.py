import numpy as np
import matplotlib.pyplot as plt


def plot_proporciones(y, conjuntos):
    """
    :param y: Columna target del conjunto de datos
    :param conjuntos: dict {nombre: índices}, por ejemplo {"Train": train_idx, "Test": test_idx}
    """
    y = np.asarray(y)
    clases = np.unique(y)

    # Fila 0 = dataset completo, para tener la referencia
    nombres = ["Dataset"] + list(conjuntos)
    grupos = [y] + [y[idx] for idx in conjuntos.values()]

    x = np.arange(len(clases))
    ancho = 0.8 / len(grupos)

    for i, (nombre, yy) in enumerate(zip(nombres, grupos)):
        props = [(yy == c).mean() for c in clases]
        plt.bar(x + i * ancho, props, ancho, label=f"{nombre} (n={len(yy)})")

    plt.xticks(x + 0.4 - ancho / 2, clases)
    plt.xlabel("Clase")
    plt.ylabel("Proporción")
    plt.legend()
    plt.show()