import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk import word_tokenize
from stanza import Pipeline as stPipeline
import stanza

from string import punctuation
import re
import unicodedata


class Transformer:

	def __init__(self):
		self.stop_words_es = set(stopwords.words('spanish'))
		# stanza.download(lang='es') #, model_dir='PYTHON_RESOURCES/stanza_resources')
		self.nlp = stPipeline('es', package='ancora', processors='tokenize,mwt,pos,lemma')
		# , lemma_model_path='PYTHON_RESOURCES/stanza_resources/es/lemma/ancora_customized.pt')

	def prepare_data(self, data):
		# self.check_null_columns(data)
		data = self.remove_null(data)
		# data = self.filter_corpus_posadas(data)
		# loader.save_to_csv(data, './in_data/development.csv')
		# print(data)
		data['article_text'] = data['article_text'].apply(lambda txt: re.sub(r'[^\wñ\s]', r'',
																	   str(txt)
																	   , 0, re.I))
		# unicodedata.normalize("NFKD", txt).encode('ascii', 'ignore').decode('utf8')
					# r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1",
		# print(data['article_text'])
		non_words = self.symbols_to_remove()
		data['article_text'] = data['article_text'].apply(lambda txt:
														  ''.join([c for word in txt for c in word if c not in non_words]))
		return data

	def lemmatization(self, col_text):

		col_text_lemma = col_text.apply(lambda txt: self.nlp(txt))
		# print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for text in col_text_lemma for sent in text.sentences for word in sent.words], sep='\n')
		col_text_lemma = col_text_lemma.apply(
			lambda doc_lemmatized: ' '.join([word.lemma for sent in doc_lemmatized.sentences for word in sent.words]))
		# print(col_text_lemma)	
		
		return col_text_lemma

	def nlp_process(self, col_text):
		doc_analized = col_text.apply(lambda txt: self.nlp(txt))
		return doc_analized

	def lemmatization_feature(self, docs_nlp):
		col_text_lemma = docs_nlp.apply(
			lambda doc: ' '.join([word.lemma for sent in doc.sentences for word in sent.words]))
		return col_text_lemma

	def pronouns_count(self, col_text):
		counts = col_text.apply(
			lambda doc: sum([1 for sent in doc.sentences for word in sent.words if word.upos == 'PRON'])
		)
		return counts

	def adjectives_count(self, col_text):
		counts = col_text.apply(
			lambda doc: sum([1 for sent in doc.sentences for word in sent.words if word.upos == 'ADJ'])
		)
		return counts

	def propernoun_count(self, col_text):
		counts = col_text.apply(
			lambda doc: sum([1 for sent in doc.sentences for word in sent.words if word.upos == 'PROPN'])
		)
		return counts

	def noun_count(self, col_text):
		counts = col_text.apply(
			lambda doc: sum([1 for sent in doc.sentences for word in sent.words if word.upos == 'NOUN'])
		)
		return counts

	def verb_count(self, col_text):
		counts = col_text.apply(
			lambda doc: sum([1 for sent in doc.sentences for word in sent.words if word.upos == 'VERB'])
		)
		return counts

	def adverb_count(self, col_text):
		counts = col_text.apply(
			lambda doc: sum([1 for sent in doc.sentences for word in sent.words if word.upos == 'ADV'])
		)
		return counts
		
	def symbols_to_remove(self):
		non_words = list(punctuation)
		non_words.extend(['¿', '¡', '*'])
		non_words.extend(map(str, range(10)))
		# non_words.remove('~')
		return non_words

	def tokenize_text(self, text):

		# Caracteres repetidos
		text = re.sub(r'(.)\1+', r'\1\1', text)

		tokens = word_tokenize(text, language='spanish')

		return tokens

	def remove_stopwords(self, txt):
		content = [w.lower() for w in txt.split() if w.lower() not in self.stop_words_es]
		return content  # len(content) / len(stop_words_es)

	def remove_null(self, df):
		# df.drop(axis=1, inplace=True)
		df = df[~df.isnull()]
		# df = df[~df['subtitle'].isnull()]
		# df = df[~df['author'].isnull()]  TEMPORALMENTE COMENTADO
		# df.drop_duplicates(subset=['url'], keep='last') # Not needed
		return df

	def check_null_columns(self, df):
		for colu in df.columns:  # ['article_text', 'author', 'date_time', 'location', 'section', 'subtitle', 'title', 'url']:
			counts = 0
			for field in df[colu]:
				if field is None:
					continue
				if type(field) == str:
					counts = counts + 1
			# elif not np.isnan(field):
			# counts = counts + 1
			print(colu, counts)

	def filter_corpus_posadas(self, df):
		corpus = df[df['Topic'] == 'Politics']
		corpus = corpus.drop(['Id', 'Topic'], axis=1)
		corpus.rename(columns={'Text': 'article_text', 'Headline': 'subtitle'}, inplace=True)
		corpus['Category'] = corpus['Category'].apply(lambda tag: 1 if tag == 'Fake' else 0)
		corpus = corpus.astype({"article_text": str, "subtitle": str, "Source": str, "Link": str})
		return corpus
