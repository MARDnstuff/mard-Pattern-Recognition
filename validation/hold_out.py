from sampling.stratified import stratified_holdout
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)



def hold_out(data: pd.DataFrame):
    """
    :param data: El conjunto completo de datos
    """

    train_idx, val_idx = stratified_holdout(data.y, test_size=0.4, random_state=42) 
    X = data.X.to_numpy()
    y = data.y.to_numpy().ravel()

    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_val)

    scores = accuracy_score(y_val, y_pred)

    logger.info(f"Accurancy score: {scores} ")
    return scores