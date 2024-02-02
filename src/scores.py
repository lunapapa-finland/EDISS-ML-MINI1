# import necessary Libs
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ignore warning
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime

# Set up logging configuration
log_filename = 'results/scores.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load data
    df_preprocessed = pd.read_csv('data/preprocessed.csv')  

    # Separate Features from Label
    X = df_preprocessed.iloc[:, :-1]
    y = df_preprocessed.iloc[:, -1:]
    y = y.values.ravel()

    # Apply MinMaxScaler on numeric features
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # Get model scores
    scores = get_scores(X,y)

    # Convert the dictionary to a list of lists for seaborn boxplot
    data = [scores[label] for label in scores.keys()]

    # Set seaborn style
    sns.set(style="whitegrid")

    # Plot model performance for comparison using Seaborn
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(data=data, showmeans=True)
    ax.set_xticklabels(list(scores.keys()), rotation=45)
    ax.set_ylabel('Performance')

    # Save the plot as an image
    plt.savefig('results/performance_comparison.png')
    
    

# define a model space using dictionary
def get_models():
  names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "GradientBoosting",
    "SGD",
    "Perceptron",
  ]
  classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", random_state=42, class_weight='balanced', C=0.025),
    SVC(gamma=2, C=1, random_state=42, class_weight='balanced'),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=4),
    DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42, class_weight='balanced'),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier(random_state=42),
    SGDClassifier(random_state=42, class_weight='balanced'),
    Perceptron(random_state=42)
  ]
  return dict(zip(names,classifiers))

def get_scores(X,y):
  scores = {key: None for key in get_models().keys()}
  cv = RepeatedStratifiedKFold(n_splits=2,
                                 n_repeats=5,
                                 random_state=1)
  for name, claasifier in (get_models().items()):
      scores[name] = cross_val_score(claasifier, X, y, cv=cv, scoring='accuracy', n_jobs=4)
      logging.info(f'{name} model is done.')
  return scores


if __name__ == "__main__":

    logging.info('Start!')
    main()
