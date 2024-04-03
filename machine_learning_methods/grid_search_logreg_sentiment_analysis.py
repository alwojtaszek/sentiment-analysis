from sklearn.linear_model import LogisticRegression

from data_preperation.hybrid_word2vec_tfidf_data_prep import x_train, y_train, y_test, x_test
from evaluation.evaluation_metrics import calculate_evaluation_metrics
from machine_learning_methods.grid_search.logreg_grid_search import perform_grid_search


def train_evaluate_sentiment_model(x_train, y_train, x_test, y_test):
    """
    Trains a logistic regression model for sentiment analysis and evaluates its performance on a test dataset.

    This function trains a logistic regression model using a specified regularization strength and penalty.
    It leverages a hybrid feature extraction method combining Word2Vec and TF-IDF. The model's performance
    is evaluated using accuracy, precision, recall, and F1 score metrics.

    Parameters:
    - x_train (array-like): Training features.
    - y_train (array-like): Training labels.
    - x_test (array-like): Testing features.
    - y_test (array-like): Testing labels.

    Returns:
    - dict: A dictionary containing the model's performance metrics ('accuracy', 'precision', 'recall', 'f1_score').
    """

    # Initialize and fit the logistic regression model
    best_params = perform_grid_search(x_train, y_train)
    lr_model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'],
                                  solver=best_params['solver']).fit(x_train, y_train)

    # Predict on the test set
    y_pred = lr_model.predict(x_test)

    # Calculate performance metrics
    metrics = calculate_evaluation_metrics(y_test, y_pred)

    return metrics


if __name__ == "__main__":
    train_evaluate_sentiment_model(x_train, y_train, x_test, y_test)
