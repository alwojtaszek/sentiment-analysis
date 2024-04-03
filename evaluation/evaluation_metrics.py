from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_evaluation_metrics(y_test, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    # Print performance metrics
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
