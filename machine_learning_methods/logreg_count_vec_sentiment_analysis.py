from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data_preperation.data_prep import get_data, preprocess
from evaluation.evaluation_metrics import calculate_evaluation_metrics
from machine_learning_methods.grid_search.logreg_grid_search import perform_grid_search


def train_evaluate_sentiment_model():
    data = get_data()
    sentences = data['comment'].apply(lambda x: ' '.join(preprocess(x)))

    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    bow = bow_vectorizer.fit_transform(sentences)

    X_train, X_test, y_train, y_test = train_test_split(bow, data['sentiment'].values,
                                                        test_size=0.33, random_state=42, stratify=data['sentiment'])

    best_params = perform_grid_search(X_train, y_train)
    model = LogisticRegression(solver=best_params['solver'],
                               penalty=best_params['penalty'], C=best_params['C'])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    calculate_evaluation_metrics(y_test, y_pred)


if __name__ == "__main__":
    train_evaluate_sentiment_model()

# Accuracy: 0.8533
# Precision: 0.8555
# Recall: 0.8533
# F1_score: 0.8519
