# import necessary Libs

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
# ignore warning
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime

# Set up logging configuration
log_filename = 'results/KNN.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load data
    df_preprocessed = pd.read_csv('data/preprocessed.csv')  

    # Separate Features from Label
    X = df_preprocessed.iloc[:, :-1]
    y = df_preprocessed.iloc[:, -1:]
    y = y.values.ravel()
    # Define the KNeighborsClassifier
    knn = KNeighborsClassifier()

    # Define the parameter grid to search
    param_grid = {
        'n_neighbors': np.arange(1, 21),  # Range from 1 to 20
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }

    # Create a grid search object
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='accuracy', cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42), n_jobs=4)

    # Fit the grid search to the data
    grid_search.fit(X, y)

    # Extract results
    results = grid_search.cv_results_
    n_neighbors_values = param_grid['n_neighbors']
    uniform_scores_p1 = results['mean_test_score'][(results['param_weights'] == 'uniform') & (results['param_p'] == 1)]
    uniform_scores_p2 = results['mean_test_score'][(results['param_weights'] == 'uniform') & (results['param_p'] == 2)]
    distance_scores_p1 = results['mean_test_score'][(results['param_weights'] == 'distance') & (results['param_p'] == 1)]
    distance_scores_p2 = results['mean_test_score'][(results['param_weights'] == 'distance') & (results['param_p'] == 2)]

    # Plot the results in a 2x2 grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    axes[0, 0].plot(n_neighbors_values, uniform_scores_p1, label='Uniform Weights, p=1', marker='o')
    axes[0, 0].set_title('Uniform Weights, p=1')

    axes[0, 1].plot(n_neighbors_values, uniform_scores_p2, label='Uniform Weights, p=2', marker='o')
    axes[0, 1].set_title('Uniform Weights, p=2')

    axes[1, 0].plot(n_neighbors_values, distance_scores_p1, label='Distance Weights, p=1', marker='o')
    axes[1, 0].set_title('Distance Weights, p=1')

    axes[1, 1].plot(n_neighbors_values, distance_scores_p2, label='Distance Weights, p=2', marker='o')
    axes[1, 1].set_title('Distance Weights, p=2')

    # Set common labels and legend
    for ax in axes.flat:
        ax.set(xlabel='Number of Neighbors (k)', ylabel='Mean Test Score (Accuracy)')
        ax.legend()

    # Adjust layout
    plt.tight_layout()
   

    # Save the plot as an image
    plt.savefig('results/KNN.png')
    # Print the best parameters and corresponding accuracy
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    logging.info("Best Parameters: %s", best_params)
    logging.info("Best Accuracy: %s", best_accuracy)




if __name__ == "__main__":
   
    logging.info('Start!')
    main()
