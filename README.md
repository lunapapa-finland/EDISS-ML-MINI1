
# Machine Learning Project README

This project involves building and evaluating machine learning models using various algorithms. It includes scripts for generating cross validation scores using base mdels and fine tuning models of  k-Nearest Neighbors (KNN) model, and Adaboost model using grid search.

## Directory Structure

```
project-root/
│
├── src/
│   ├── scores.py
│   ├── KNN.py
│   ├── Adaboost.py
│   └── ...
├── results/
│   ├── performance_comparison.png
│   ├── KNN.png
│   ├── Adaboost.png
│   ├── scores.log
│   ├── KNN.log
│   ├── Adaboost.log
│   └── ...
├── data/
│   ├── bank-additional-full.csv
│   └── preprocessed.csv
├── requirements.txt
├── Makefile
├── README.md
└── ...
```

- **src/:** Contains Python scripts for generating scores, implementing KNN, Adaboost, and other related functionalities.
- **results/:** Stores log files for each script execution and Holds result files, such as performance comparison plots.
- **data/:** Includes the preprocessed dataset in CSV format.
- **requirements.txt:** Lists project dependencies.
- **Makefile:** Defines commands for running scripts.

## Logging

Logging is implemented using the Python `logging` module. Log files are stored in the `results/` directory. The logs capture information about the execution flow, successful runs, and encountered errors.

## Makefile Commands

- **scores:** Generates scores using the `scores.py` script.
- **KNN:** Implements the k-Nearest Neighbors model using the `KNN.py` script.
- **Adaboost:** Implements the Adaboost model using the `Adaboost.py` script.

Example usage:

```bash
make scores
make KNN
make Adaboost
```

## Prerequisites(recommend to use Conda)

Ensure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```
