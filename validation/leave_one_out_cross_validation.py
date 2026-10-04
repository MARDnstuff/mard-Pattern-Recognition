from sampling.stratified import stratified_kfold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def leave_one_out(data: pd.DataFrame):
    """
    :param data: El conjunto completo de datos
    """
    X = data.X.to_numpy()
    y = data.y.to_numpy().ravel()

    scores = []
    for idx, row in enumerate(X):

        X_train = np.concatenate([X[:idx], X[idx+1:]])
        y_train = np.concatenate([y[:idx], y[idx+1:]])
        X_val = X[idx:idx+1]
        y_val = y[idx:idx+1]

        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))


        model.fit(X_train,y_train)

        y_pred = model.predict(X_val)

        scores.append(accuracy_score(y_val, y_pred))

    sc = np.mean(scores)
    logger.info(f"Accurancy score: {sc} ")
    return sc






