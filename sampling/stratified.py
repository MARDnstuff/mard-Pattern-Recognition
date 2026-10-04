from utils.utils import plot_proporciones
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def stratified_kfold(y, k: int = 5, random_state: int = 42 ):
    y = np.asarray(y)
    if k < 2:
        raise ValueError("k debe ser al menos 2")

    rng = np.random.default_rng(random_state)

    # folds[i] acumulará un trozo de cada clase
    folds = [[] for _ in range(k)]
    offset = 0  # dónde empiezan a repartirse los sobrantes en la clase actual

    for c in np.unique(y):
        # 1. Índices de la clase y barajado
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)

        # 2. Tamaño del trozo para cada fold
        n = len(idx)
        sizes = np.full(k, n // k)
        sizes[: n % k] += 1               # reparte los sobrantes
        sizes = np.roll(sizes, offset)    # rota quién recibe los extras
        offset = (offset + n % k) % k     # la siguiente clase sigue donde quedó esta

        # 3. Cortar la clase en k trozos disjuntos
        cortes = np.cumsum(sizes)[:-1]
        for i, trozo in enumerate(np.split(idx, cortes)):
            folds[i].append(trozo)

    # 4. Unir los trozos de cada fold y barajar
    folds = [rng.permutation(np.concatenate(f)) for f in folds]

    # 5. En la iteración i: fold i = validación, el resto = train
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((rng.permutation(train_idx), val_idx))

    plot_proporciones(y, {f"Fold {i}": val for i, (_, val) in enumerate(splits)})
    
    return splits


def stratified_holdout(y,  test_size: float = 0.2, random_state: int = 42):
    """
    
    :para y: Columna target de nuestro conjunto de datos

    """
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    train_idx, test_idx = [], []


    # Para cada clase unica en 'y'
    for c in np.unique(y):

        # Lista de indices de las instancias que pertenecen a la clase c
        idx = np.flatnonzero(y == c)
        # Randomiza
        rng.shuffle(idx)

        n = len(idx)
        # Indica cuanto de nuestro idx va ir a test
        n_test = int(round(n * test_size)) # round va al par más cercano


        # con esto evitamos que n_test sea 0
        if n >= 2:
            n_test = min(max(n_test, 1), n - 1)
        else:
            # si solo hay una muesta, la mandamos a test
            n_test = 0

        # Se hace la separación
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])

    # Une las listas y las vuelve a mezclar
    train_idx = rng.permutation(np.concatenate(train_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))

    plot_proporciones(y, {"Train": train_idx, "Test": test_idx})
    return train_idx, test_idx

