from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV

from data_preperation.hybrid_word2vec_tfidf_data_prep import x_train, y_train


def perform_grid_search(x_train, y_train):
    """
    Performs grid search to find the best hyperparameters for a logistic regression model
    on sentiment analysis data. The search includes variations in penalty type, regularization
    strength (C), and solver type.

    The function uses a logistic regression classifier and tests combinations of hyperparameters
    defined in a parameter grid. The best combination of parameters is determined based on cross-validated
    performance on the training data.

    Parameters:
    - x_train (array-like): Training features, processed through a hybrid Word2Vec and TF-IDF pipeline.
    - y_train (array-like): Training labels, corresponding to sentiment categories.

    Returns:
    - dict: The best hyperparameters found during the grid search.
    """
    classifier = LogisticRegression(random_state=42)
    param_grid = [
        {
            'penalty': ['l1', 'l2'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['liblinear']
        }
    ]

    clf = GridSearchCV(classifier, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_clf = clf.fit(x_train, y_train)

    print("Best hyperparameters found:")
    for param, value in best_clf.best_params_.items():
        print(f"{param}: {value}")

    return best_clf.best_params_


# Example usage
if __name__ == "__main__":
    best_params = perform_grid_search(x_train, y_train)
    print(best_params)
