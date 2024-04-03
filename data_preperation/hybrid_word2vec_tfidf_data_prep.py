import numpy as np
import gensim
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_preperation.data_prep import get_data, preprocess

"""
Performs sentiment analysis using logistic regression, leveraging both Word2Vec and TF-IDF vectors.
A sentiment analysis dataset from Kaggle is required for training and testing.
"""

# Word2Vec model
word2vec = gensim.models.Word2Vec(vector_size=100, window=3, min_count=2, sg=1)

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words='english')


def embed_sentence(sentence, word2vec_mdl):
    """
    Averages Word2Vec vectors of the words in a sentence to create a single vector.

    Parameters:
    - sentence (str): The sentence to convert.

    Returns:
    - ndarray: The averaged vector of the sentence.
    """
    vecs = [word2vec_mdl.wv[word] for word in sentence if word in word2vec_mdl.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(word2vec_mdl.vector_size)


data = get_data()
sentences = data['comment'].apply(lambda x: ' '.join(preprocess(x)))

tfidf_features = tfidf_vectorizer.fit_transform(sentences)
word2vec.build_vocab(sentences)

# Feature engineering
sentence_vectors = np.array([embed_sentence(s.split(), word2vec_mdl=word2vec) for s in sentences])
word2vec_sparse = csr_matrix(sentence_vectors)

combined_features = hstack([tfidf_features, word2vec_sparse])

# Encoding sentiment labels and splitting the dataset
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['sentiment'])

x_train, x_test, y_train, y_test = train_test_split(combined_features, y,
                                                    test_size=0.33, random_state=42, stratify=y)

