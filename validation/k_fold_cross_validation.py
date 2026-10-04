from sampling.stratified import stratified_kfold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def k_fold_cv(data: pd.DataFrame):
    """
    :param data: El conjunto completo de datos
    """

    splits = stratified_kfold(data.y, k=5, random_state=42) 
    scores = []
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    for train_idx, val_idx in splits:

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

        model.fit(X_train,y_train)

        y_pred = model.predict(X_val)

        scores.append(accuracy_score(y_val, y_pred))

    sc = np.mean(scores)
    logger.info(f"Accurancy score: {sc} ")
    return sc



