import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RedditBitcoinSentiment():
    """
    Source: https://www.kaggle.com/datasets/shisha01/reddit-crypto-sentiment-and-market-trend-dataset
    """
    def __init__(self):
        # fetch dataset
        self.file_path = "data/datasets/finbert_scores_added.csv"
        self.df =  pd.read_csv(self.file_path)

        self.df.dropna()

        # df = data (as pandas dataframes) 
        targets = ["class_1h"]

        columnas_numericas = [
            col for col in self.df.columns
            if pd.to_numeric(self.df[col], errors='coerce').notna().all()
        ]

        self.y = self.df[targets]
        self.X = self.df[columnas_numericas] 

        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.dropna()
        self.y = self.y.loc[self.X.index]

        logger.debug(self.df.head) 

