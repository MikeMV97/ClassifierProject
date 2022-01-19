import pandas as pd
import numpy as np
from IPython.display import display
import time
from featuresHelper import FeaturesHelper
from textTransformer import Transformer
from load import Loader


def backwardElimination(x, Y, sl, columns):
    ini = len(columns)
    numVars = x.shape[1]
    for i in range(0, numVars):
        regressor = sm.OLS(Y, x).fit()
        maxVar = max(regressor.pvalues) #.astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor.pvalues[j].astype(float) == maxVar):
                    columns = np.delete(columns, j)
                    x = x.loc[:, columns]
                    
    print('\nSelect {:d} features from {:d} by best p-values.'.format(len(columns), ini))
    print('The max p-value from the features selecte is {:.3f}.'.format(maxVar))
    print(regressor.summary())
    
    # odds ratios and 95% CI
    conf = np.exp(regressor.conf_int())
    conf['Odds Ratios'] = np.exp(regressor.params)
    conf.columns = ['2.5%', '97.5%', 'Odds Ratios']
    display(conf)
    
    return columns, regressor


def test_pvalues(df, y_train):
    featuresHelper = FeaturesHelper()
    pv_cols = featuresHelper.columns_numeric.values
    SL = 0.051
    pv_cols, LR = backwardElimination(df, y_train, SL, pv_cols)
    print('pv_cols:', pv_cols)
    print('Linear Regressor:', LR)


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
    
    # start = time.time()
    # text['rate_pron'] = np.divide(text['pron_counts'] , text['num_words'])
    # text['rate_adj'] = np.divide(text['adj_counts'] , text['num_words'])
    # text['rate_adv'] = np.divide(text['adv_counts'] , text['num_words'])
    # text['rate_noun'] = np.divide(text['noun_counts'] , text['num_words'])
    # text['rate_verb'] = np.divide(text['verb_counts'] , text['num_words'])
    # text['rate_propn'] = np.divide(text['propn_counts'], text['num_words'])
    # end = time.time()
    # print('Division time with numpy:', end - start)
    # # On several tries, Pandas was at least x5 times faster than numpy in this case
    # start = time.time()
    text['rate_pron'] = text['pron_counts'] / text['num_words']
    text['rate_adj'] = text['adj_counts'] / text['num_words']
    text['rate_adv'] = text['adv_counts'] / text['num_words']
    text['rate_noun'] = text['noun_counts'] / text['num_words']
    text['rate_verb'] = text['verb_counts'] / text['num_words']
    text['rate_propn'] = text['propn_counts'] / text['num_words']
    # end = time.time()
    # print('Division time with pandas:', end - start)
    
    # print( text['txt_lemmatized'] )
    # print( text['pron_counts'])
    # print( text['adj_counts'])
    # print( text['adv_counts'])
    # print( text['noun_counts'])
    # print( text['verb_counts'])
    # print( text['propn_counts'])
    # print( '----------------TASAS:--------------------' )
    # print( text['rate_pron'])
    # print( text['rate_adj'])
    # print( text['rate_adv'])
    # print( text['rate_noun'])
    # print( text['rate_verb'])
    # print( text['rate_propn'])
    # text['rate_propn'] = text['propn_counts'] / text['num_words']
    # print( text['rate_propn'])
    # print( '----------------Palabras:--------------------' )
    # print( text['num_words'])

    y_labels = data['Category']
    test_pvalues(df=text, y_train=y_labels)

