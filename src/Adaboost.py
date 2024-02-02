# import necessary Libs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
# ignore warning
import warnings

import logging
from datetime import datetime

# Set up logging configuration
log_filename = 'results/Adaboost.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Load data
    df_preprocessed = pd.read_csv('data/preprocessed.csv')  

    # Separate Features from Label
    X = df_preprocessed.iloc[:, :-1]
    y = df_preprocessed.iloc[:, -1:]
    y = y.values.ravel()
    # Define the base decision tree estimator
    base_estimator = DecisionTreeClassifier()

    # Define the AdaBoostClassifier with the base_estimator
    adaboost = AdaBoostClassifier(estimator=base_estimator)

    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R'],
        'base_estimator__max_depth': [1, 2, 3],  
        'base_estimator__min_samples_split': [2, 5, 10] 
    }

    # Create a grid search object
    grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, scoring='accuracy', cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42), n_jobs=4)

    # Fit the grid search to the data
    grid_search.fit(X, y)

    # Extract results
    results = grid_search.cv_results_
    n_estimators_values = param_grid['n_estimators']
    learning_rate_values = param_grid['learning_rate']

    # Plot the results in a 2x2 grid
    fig, axes = plt.subplots(nrows=len(learning_rate_values), ncols=len(n_estimators_values), figsize=(15, 10), sharex=True, sharey=True)

    for i, learning_rate in enumerate(learning_rate_values):
        for j, n_estimators in enumerate(n_estimators_values):
            scores = results['mean_test_score'][(results['param_n_estimators'] == n_estimators) & (results['param_learning_rate'] == learning_rate)]
            axes[i, j].plot(scores, marker='o')
            axes[i, j].set_title(f'Learning Rate={learning_rate}, n_estimators={n_estimators}')

    # Set common labels
    for ax in axes.flat:
        ax.set(xlabel='Iterations', ylabel='Mean Test Score (Accuracy)')

    # Adjust layout
    plt.tight_layout()
   

    # Save the plot as an image
    plt.savefig('results/Adaboost.png')
    # Print the best parameters and corresponding accuracy
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    logging.info("Best Parameters: %s", best_params)
    logging.info("Best Accuracy: %s", best_accuracy)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.info('Start!')
        main()
