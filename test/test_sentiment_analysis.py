from io import StringIO

import gensim
import numpy as np
from numpy import float32, array
import pandas as pd

from data_preperation.hybrid_word2vec_tfidf_data_prep import preprocess, embed_sentence


def test_preprocess():
    data = """i didnt feel humiliated;sadness
i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake;sadness
im grabbing a minute to post i feel greedy wrong;anger"""
    pd_df = pd.read_csv(StringIO(data), delimiter=';', names=['comment', 'sentiment'])
    clean_words = pd_df['comment'].apply(preprocess)

    assert clean_words[0] == ['feel', 'humiliate']
    assert clean_words[1] == ['go', 'feel', 'hopeless', 'damn', 'hopeful', 'around', 'someone', 'care', 'awake']
    assert clean_words[2] == ['grab', 'minute', 'post', 'feel', 'greedy', 'wrong']


def test_word2vec():
    pd_df = pd.DataFrame({'comment': [['feel', 'humiliate'],
                                      ['go', 'feel', 'hopeless', 'damn', 'hopeful', 'around', 'someone', 'care',
                                       'awake'],
                                      ['grab', 'minute', 'post', 'feel', 'greedy', 'damn', 'wrong']]})
    model = gensim.models.Word2Vec(min_count=2)
    model.build_vocab(pd_df['comment'])
    text_vectors = [embed_sentence(s, model) for s in pd_df['comment']]
    assert len(text_vectors[0]) == 100
    assert len(text_vectors[1]) == 100
    assert len(text_vectors[2]) == 100



