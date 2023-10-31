import matplotlib.pyplot as plt
import pandas as pd


def corr_scatter(df: pd.DataFrame, target: str):
    # get column names from df
    # iterate through all columns and plot scatterplot for all columns
    column_names = df.columns
    print(column_names)