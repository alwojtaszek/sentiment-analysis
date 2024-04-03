import string

import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize WordNet lemmatizer
wnl = WordNetLemmatizer()

# Define a comprehensive set of English stopwords
stop_words = set(stopwords.words('english')).union({
    "didnt", "dont", "cant", "wont", "im", "ive", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
    "hadnt", "doesnt", "couldnt", "shouldnt", "mightnt", "mustnt"
})


def get_data(url=r'C:\Users\alwoj\PycharmProjects\sentiment-analysis\data\*.txt'):
    """
    Loads text data from files.

    Parameters:
    - url (str): The path to the data file.

    Returns:
    - DataFrame: The loaded data.
    """
    import glob
    data = [pd.read_csv(filename, delimiter=';', names=['comment', 'sentiment']) for filename in glob.glob(url)]
    return pd.concat(data, axis=0)


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
