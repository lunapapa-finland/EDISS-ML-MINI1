# import necessary Libs
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
# ignore warning
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime

# Set up logging configuration
log_filename = 'results/preprocessing.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load data
    df = pd.read_csv('data/bank-additional-full.csv', sep=';')  
    unique_values_per_column = df.nunique()
    logging.info(unique_values_per_column)
    logging.info(df.info())
    # Identify the categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Apply one-hot encoding with drop_first=True to all categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    df_combined = df_encoded.copy()
    for cat_column in categorical_columns:
      original_column_values = df[cat_column].unique()
      encoded_column_names = [f"{cat_column}_{value}" for value in original_column_values]
      logging.info(f" OriginalColumn: {cat_column}")
      logging.info(f"Encoded Columns: {encoded_column_names}")
    # Calculate the correlation matrix
    corr_matrix = df_combined.corr()

    # Set the correlation threshold
    correlation_threshold = 0.1

    # Create a mask to hide the upper triangle of the correlation matrix
    mask = (corr_matrix.abs() < correlation_threshold) & (corr_matrix.abs() != 1)

    # Apply the mask to hide values below the threshold
    masked_corr_matrix = corr_matrix.mask(mask)

    # Find the indices of rows with 'y_yes' values lower than the threshold
    low_corr_column = masked_corr_matrix[masked_corr_matrix['y_yes'].isna()].index

    # Drop features from corr
    corr_matrix.drop(low_corr_column, axis=1, inplace=True)
    corr_matrix.drop(low_corr_column, inplace=True)

    # Set the size of the plot
    plt.figure(figsize=(12, 10))

    # Create the heatmap with adjusted font size
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", vmin=-1, vmax=1,
                annot_kws={"size": 8})

    plt.title("Correlation Matrix Heatmap")
    plt.savefig('results/Heatmap.png')
    # Drop the low_corr_column features from encoded df according to the threshold
    df_preprocessed = df_combined.drop(low_corr_column, axis = 1)
    # Drop extra feature 'duration' from encoded df since:
    # this attribute highly affects the output target 
    # (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. 
    # Also, after thea end of the call, y is known. Thus, this input should only be included for benchmark purposes 
    # and should be discarded if the intention is to hve a realistic predictive model.
    df_preprocessed.drop('duration', axis = 1, inplace = True)
    # Print the remaining features for traninig model: 
    logging.info(f'features:{df_preprocessed.columns[:-1].tolist()}')
    # Print the remaining features for traninig model: 
    logging.info(f'label: {df_preprocessed.columns[-1:].tolist()}')
    #save to csv for further usage especially in dedicated machine in the following section
    #(#chart-the-accuracy-using-predefined-configurations-for-various-models)
    df_preprocessed.to_csv('data/preprocessed.csv', index=False)
    
if __name__ == "__main__":

    logging.info('Start!')
    main()
