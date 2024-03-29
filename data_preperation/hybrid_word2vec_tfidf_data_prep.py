import string
import numpy as np
import pandas as pd
import gensim
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

"""
Performs sentiment analysis using logistic regression, leveraging both Word2Vec and TF-IDF vectors.
A sentiment analysis dataset from Kaggle is required for training and testing.
"""

from nltk.corpus import stopwords

# Initialize WordNet lemmatizer
wnl = WordNetLemmatizer()
# Word2Vec model
word2vec = gensim.models.Word2Vec(vector_size=100, window=3, min_count=2, sg=1)

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words='english')

# Define a comprehensive set of English stopwords
stop_words = set(stopwords.words('english'))

def get_data(url='data/train.txt'):
    """
    Loads text data from a file.

    Parameters:
    - url (str): The path to the data file.

    Returns:
    - DataFrame: The loaded data.
    """
    return pd.read_csv(url, delimiter=';', names=['comment', 'sentiment'])


def preprocess(text):
    """
    Prepares a text by converting to lowercase, removing punctuation,
    excluding stopwords, and lemmatizing the words.

    Parameters:
    - text (str): The text to preprocess.

    Returns:
    - list: Preprocessed tokens.
    """
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [wnl.lemmatize(token, pos='v') for token in tokens]
    return tokens


def embed_sentence(sentence):
    """
    Averages Word2Vec vectors of the words in a sentence to create a single vector.

    Parameters:
    - sentence (str): The sentence to convert.

    Returns:
    - ndarray: The averaged vector of the sentence.
    """
    vecs = [word2vec.wv[word] for word in sentence if word in word2vec.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(word2vec.vector_size)


# Data loading and preprocessing
train, test = get_data('data/train.txt'), get_data('data/test.txt')
data = pd.concat([train, test])
sentences = data['comment'].apply(lambda x: ' '.join(preprocess(x)))

tfidf_features = tfidf_vectorizer.fit_transform(sentences)
word2vec.build_vocab(sentences)

# Feature engineering
sentence_vectors = np.array([embed_sentence(s.split()) for s in sentences])
word2vec_sparse = csr_matrix(sentence_vectors)

combined_features = hstack([tfidf_features, word2vec_sparse])

# Encoding sentiment labels and splitting the dataset
label_encoder = LabelEncoder()
y_train, y_test = label_encoder.fit_transform(train['sentiment']), label_encoder.transform(test['sentiment'])
x_train, x_test = combined_features[:len(train)], combined_features[len(train):]


