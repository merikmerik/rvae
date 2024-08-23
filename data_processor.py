# prepare the dataframe for use

import pandas as pd

def load_data():
    dataframe_4 = pd.read_pickle("dataframe_4.pkl")
    return dataframe_4