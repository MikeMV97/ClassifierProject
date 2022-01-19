import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk import tokenize
from nltk import bigrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sentiment_analysis_spanish import sentiment_analysis
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from textTransformer import Transformer


class FeaturesHelper:
	def __init__(self):
		self.columns_numeric = ['num_words', 'num_diff_words', 'num_stopwords', 'avg_word_len', 'rate_stopwords_words',
						   'rate_diffwords_words', 'rate_pron', 'rate_adj', 'rate_adv', 'rate_noun', 'rate_verb', 
						   'rate_propn', 'sentiment_txt']

	def add_features(self, data_txt):
		transformer = Transformer()
		stop_words_es = set(stopwords.words('spanish'))

		data_feat = pd.DataFrame(data_txt, columns=['article_text'])		# df.rename(columns={"A": "a"})
		data_feat['article_text'] = transformer.prepare_data(data_feat)

		# PROCESS Lemmatization of article text
		data_feat['txt_nlp']     = transformer.nlp_process(data_feat['article_text'])
		data_feat['article_text']   = transformer.lemmatization_feature(data_feat['txt_nlp'])
		# data_feat['article_text'] = transformer.lemmatization(data_feat['article_text'])

		# GET average word length
		data_feat['avg_word_len'] = data_feat['article_text'].apply(
			lambda x: np.mean([len(t) for t in x.split() if t not in stop_words_es]))
			# if len([len(t) for t in x.split() if t not in stop_words_es]) > 0 else 0
		data_feat['avg_word_len'] = data_feat['avg_word_len'].astype('float32')

		data_feat['sentiment_txt'] = self.get_sentiment_analysis(data_feat['article_text'])
		data_feat['sentiment_txt'] = data_feat['sentiment_txt'].astype('float32')

		# data_feat['word_count'] = get_word_count(data_feat['article_text_nosw'])

		# GET number of words in each message:
		data_feat['num_words'] = data_feat['article_text'].apply(lambda x: len(x.split()))
		data_feat['num_words'] = data_feat['num_words'].astype('int32')

		# GET number of words in each message:
		data_feat['num_diff_words'] = data_feat['article_text'].apply(lambda x: len(set(x.split())))
		data_feat['num_diff_words'] = data_feat['num_diff_words'].astype('int32')

		# GET the number of non stopwords in each message:
		data_feat['num_stopwords'] = data_feat['article_text'].apply(
			lambda txt: len([word for word in txt.split() if word in stop_words_es]))
		data_feat['num_stopwords'] = data_feat['num_stopwords'].astype('int32')

		# COUNTS of Part-Of-Speech & Morphological Features:
		data_feat['pron_counts'] = transformer.pronouns_count(data_feat['txt_nlp'])
		data_feat['adj_counts']  = transformer.adjectives_count(data_feat['txt_nlp'])
		data_feat['adv_counts']  = transformer.adverb_count(data_feat['txt_nlp'])
		data_feat['noun_counts'] = transformer.noun_count(data_feat['txt_nlp'])
		data_feat['verb_counts'] = transformer.verb_count(data_feat['txt_nlp'])
		data_feat['propn_counts'] = transformer.propernoun_count(data_feat['txt_nlp'])

		# RATES of Counts' features:
		data_feat['rate_stopwords_words'] = data_feat['num_stopwords'] / data_feat['num_words']
		data_feat['rate_diffwords_words'] = data_feat['num_diff_words'] / data_feat['num_words']
		data_feat['rate_pron'] = data_feat['pron_counts'] / data_feat['num_words']
		data_feat['rate_adj'] = data_feat['adj_counts'] / data_feat['num_words']
		data_feat['rate_adv'] = data_feat['adv_counts'] / data_feat['num_words']
		data_feat['rate_noun'] = data_feat['noun_counts'] / data_feat['num_words']
		data_feat['rate_verb'] = data_feat['verb_counts'] / data_feat['num_words']
		data_feat['rate_propn'] = data_feat['propn_counts'] / data_feat['num_words']
		data_feat['rate_stopwords_words'] = data_feat['rate_stopwords_words'].astype('float32')
		data_feat['rate_diffwords_words'] = data_feat['rate_diffwords_words'].astype('float32')
		data_feat['rate_pron'] = data_feat['rate_pron'].astype('float32')
		data_feat['rate_adj'] = data_feat['rate_adj'].astype('float32')
		data_feat['rate_adv'] = data_feat['rate_adv'].astype('float32')
		data_feat['rate_noun'] = data_feat['rate_noun'].astype('float32')
		data_feat['rate_verb'] = data_feat['rate_verb'].astype('float32')
		data_feat['rate_propn'] = data_feat['rate_propn'].astype('float32')

		# data_feat['tfidf_txt'] = get_tfidf(data_feat['article_text_sp']) # Now this is part of a Pipeline
		# print(data_feat['tfidf_txt'])

		# REMOVE stop_words_es:
		data_feat['article_text'] = data_feat['article_text'].apply(lambda txt: ' '.join(transformer.remove_stopwords(txt)))
		# data_feat.drop(columns=['article_text_sp'], inplace=True)
		data_feat.drop(columns=['pron_counts', 'adj_counts', 'adv_counts', 'noun_counts', 'verb_counts', 'propn_counts'], inplace=True)
		# print(data_feat['article_text'])

		return data_feat

	def plot_distr_cols(self, data):
		fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(9, 5.5))
		axes = axes.flat

		print( min(data['sentiment_txt']), max(data['sentiment_txt']))
		for i, column in enumerate(self.columns_numeric):
			# print( i , column)
			if column == 'sentiment_txt':
				i += 1
				ax = sns.histplot(
					data=np.log(data[self.columns_numeric]),
					x=column,
					stat="count",
					kde=True,
					color=(list(plt.rcParams['axes.prop_cycle']) * 2)[i]["color"],
					line_kws={'linewidth': 1.5},
					alpha=0.3,
					ax=axes[i]
				)
				ax.set(ylabel='Conteo')
			else:
				ax = sns.histplot(
					data=data,
					x=column,
					stat="count",
					kde=True,
					color=(list(plt.rcParams['axes.prop_cycle']) * 2)[i]["color"],
					line_kws={'linewidth': 1.5},
					alpha=0.3,
					ax=axes[i]
				)
				ax.set(ylabel='Conteo')
			axes[i].set_title(column, fontsize=7, fontweight="bold")
			axes[i].tick_params(labelsize=6)
			axes[i].set_xlabel("")
		for i in [12, 14]:
			fig.delaxes(axes[i])
		fig.tight_layout()
		plt.subplots_adjust(top=0.9)
		fig.suptitle('Distribución variables numéricas', fontsize=10, fontweight="bold")
		plt.show()

	def plot_distr_corr(self, data, y):
		fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(8, 5))
		axes = axes.flat

		for i, column in enumerate(self.columns_numeric):
			if column == 'sentiment_txt':
				i += 1
			sns.regplot(
				x=data[column],
				y=y,
				data=data,
				color="gray",
				marker='.',
				scatter_kws={"alpha": 0.4},
				line_kws={"color": "b", "alpha": 0.7},
				logistic=True,
				ax=axes[i]
			)
			axes[i].set_title(f"Category vs {column}", fontsize=7, fontweight="bold")
			# axes[i].ticklabel_format(style='sci', scilimits=(-4,4), axis='both')
			axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
			axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
			axes[i].tick_params(labelsize=6)
			# axes[i].set_xlabel("")
			axes[i].set_ylabel("Categoría")

		# Se eliminan los axes vacíos
		for i in [12, 14]:
			fig.delaxes(axes[i])

		fig.tight_layout()
		plt.subplots_adjust(top=0.9)
		fig.suptitle('Correlación con categoria', fontsize=10, fontweight="bold")
		plt.show()

	def plot_corr_matrix(self, data, annot_size: int):
		corr_matrix = data[self.columns_numeric].corr(method='pearson')
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

		sns.heatmap(
			corr_matrix,
			annot=True,
			cbar=True,
			annot_kws={"size": annot_size},
			vmin=-1,
			vmax=1,
			center=0,
			cmap=sns.diverging_palette(20, 220, n=200),
			square=True,
			ax=ax,
		)
		ax.set_xticklabels(
			ax.get_xticklabels(),
			rotation=45,
			horizontalalignment='right',
		)
		ax.tick_params(labelsize=8)
		fig.suptitle('Matriz de correlaciones', fontsize=10, fontweight="bold")
		plt.show()

	def get_tfidf(self, col_article_text):
		bigrams_txt = self.generate_bigrams(col_article_text)
		# print(*map(' '.join, bigrams_txt[0]), sep=', ')
		print(type(bigrams_txt))
		tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # min_df=2, max_df=0.8,
		features = tfidf_vectorizer.fit_transform(bigrams_txt.to_frame())

		return pd.DataFrame(features.todense(), columns=tfidf_vectorizer.get_feature_names())

	def get_sentiment_analysis(self, col_txt):
		# https://stanfordnlp.github.io/stanza/sentiment.html
		# https: // pypi.org / project / sentiment - analysis - spanish /
		sentimentalist = sentiment_analysis.SentimentAnalysisSpanish()
		sentiments = col_txt.apply(lambda txt: sentimentalist.sentiment(txt))
		# sentiments = col_txt.apply(lambda txt: 1 if sentimentalist.sentiment(txt) > 0.08 else 0)

		count = 0
		for sent in sentiments:
			if sent == 1:
				count += 1
		print(f'{count}\n')
		return sentiments

	def generate_bigrams(self, col_txt):
		tokenizer = tokenize.WhitespaceTokenizer()
		tokens = col_txt.apply(lambda txt: tokenizer.tokenize(txt))

		# print(tokens)
		# print(type(tokens[0]))
		bigramses = tokens.apply(lambda tok: bigrams(tok))
		print(type(bigramses))
		return (bigramses)

	def get_word_count(self, col_txt):
		vectorizer = CountVectorizer()
		x = vectorizer.fit_transform(col_txt)
		# print(vectorizer.get_feature_names())
		return x
