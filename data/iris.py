from ucimlrepo import fetch_ucirepo
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Iris():
    """
    Source: https://archive.ics.uci.edu/dataset/53/df
    """

    def __init__(self):
        
        # fetch dataset 
        self.df = fetch_ucirepo(id=53) 
        
        # data (as pandas dataframes) 
        self.X = self.df.data.features 
        self.y = self.df.data.targets 
        
        # metadata 
        logger.debug(self.df.metadata) 
        
        # variable information 
        logger.debug(self.df.variables) 
