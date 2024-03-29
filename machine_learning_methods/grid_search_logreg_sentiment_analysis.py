from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_preperation.hybrid_word2vec_tfidf_data_prep import x_train, y_train, y_test, x_test


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
    lr_model = LogisticRegression(C=1.623776739188721, penalty='l1', solver='liblinear').fit(x_train, y_train)

    # Predict on the test set
    y_pred = lr_model.predict(x_test)

    # Calculate performance metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    # Print performance metrics
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return metrics


# Example of using the train_evaluate_sentiment_model function
if __name__ == "__main__":
    train_evaluate_sentiment_model(x_train, y_train, x_test, y_test)
