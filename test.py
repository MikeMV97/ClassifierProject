import pandas as pd
import numpy as np
from textTransformer import Transformer
from load import Loader
import time


if __name__ == '__main__':
    # counts = []
    # suma = sum(counts)
    # print( suma )
    
    loader = Loader()
    transformer = Transformer()

    data = loader.load_from_csv('./in_data/test_data.csv')

    text = pd.DataFrame(data, columns=['article_text'])
    # print(text)
    text['article_text'] = transformer.prepare_data(text)
    # doc_proc = transformer.lemmatization(text['article_text'])
    text['num_words'] = text['article_text'].apply(lambda x: len(x.split()))

    text['txt_nlp']     = transformer.nlp_process(text['article_text'])
    text['txt_lemmatized']   = transformer.lemmatization_feature(text['txt_nlp'])
    text['pron_counts'] = transformer.pronouns_count(text['txt_nlp'])
    text['adj_counts']  = transformer.adjectives_count(text['txt_nlp'])
    text['adv_counts']  = transformer.adverb_count(text['txt_nlp'])
    text['noun_counts'] = transformer.noun_count(text['txt_nlp'])
    text['verb_counts'] = transformer.verb_count(text['txt_nlp'])
    text['propn_counts'] = transformer.propernoun_count(text['txt_nlp'])
    
    start = time.time()
    text['rate_pron'] = np.divide(text['pron_counts'] , text['num_words'])
    text['rate_adj'] = np.divide(text['adj_counts'] , text['num_words'])
    text['rate_adv'] = np.divide(text['adv_counts'] , text['num_words'])
    text['rate_noun'] = np.divide(text['noun_counts'] , text['num_words'])
    text['rate_verb'] = np.divide(text['verb_counts'] , text['num_words'])
    text['rate_propn'] = np.divide(text['propn_counts'], text['num_words'])
    end = time.time()
    print('Division time with numpy:', end - start)
    # On several tries, Pandas was at least x5 times faster than numpy in this case
    start = time.time()
    text['rate_pron'] = text['pron_counts'] / text['num_words']
    text['rate_adj'] = text['adj_counts'] / text['num_words']
    text['rate_adv'] = text['adv_counts'] / text['num_words']
    text['rate_noun'] = text['noun_counts'] / text['num_words']
    text['rate_verb'] = text['verb_counts'] / text['num_words']
    text['rate_propn'] = text['propn_counts'] / text['num_words']
    end = time.time()
    print('Division time with pandas:', end - start)
    
    # print( text['txt_lemmatized'] )
    # print( text['pron_counts'])
    # print( text['adj_counts'])
    # print( text['adv_counts'])
    # print( text['noun_counts'])
    # print( text['verb_counts'])
    print( text['propn_counts'])
    print( '----------------TASAS:--------------------' )
    # print( text['rate_pron'])
    # print( text['rate_adj'])
    # print( text['rate_adv'])
    # print( text['rate_noun'])
    # print( text['rate_verb'])
    print( text['rate_propn'])
    text['rate_propn'] = text['propn_counts'] / text['num_words']
    print( text['rate_propn'])
    print( '----------------Palabras:--------------------' )
    print( text['num_words'])

