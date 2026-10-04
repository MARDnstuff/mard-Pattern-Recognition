import pandas as pd
from ucimlrepo import fetch_ucirepo
import logging

logger = logging.getLogger(__name__)

class CreditApproval():
    """
    Source: https://archive.ics.uci.edu/dataset/27/credit+approval
    """

    # Columnas que quieres eliminar de las features
    COLS_TO_DROP = ["A13", "A12", "A10", "A9", "A7", "A6", "A5", "A4", "A1"]

    def __init__(self):
        # 1) Fetch dataset (devuelve un Bunch, NO un DataFrame)
        dataset = fetch_ucirepo(id=27)

        # 2) Separar features y target
        self.X: pd.DataFrame = dataset.data.features.copy()
        self.y: pd.Series = dataset.data.targets.squeeze()  # DataFrame → Series

        self.X = self.X.dropna()
        self.y = self.y.loc[self.X.index]

        # 3) Eliminar columnas indeseadas SOLO de las features
        self.X = self.X.drop(columns=self.COLS_TO_DROP, errors="ignore")
        
        


