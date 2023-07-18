import numpy as np
from hmmlearn import hmm
import sys
sys.path.append('..')
from models import Models

np.random.seed(42)


def build_test_model():
	# GENERATE MODEL
	num_hidden_states = 4  # Choose an appropriate number of hidden states
	model = hmm.CategoricalHMM(n_components=num_hidden_states, n_features=6, n_iter=9)

	# Convert feature vectors into a matrix
	# X = np.vstack(feature_vectors)
	# Fit the model to the data
	# model.fit(X)


def test_simple():
	# GET DATA
	train_df, test_df = retrive_featdata()
	print_info(train_df)
	train_df_outlier = train_df[ train_df['num_stopwords'] > 550 ]
	# Sort the DataFrame by the 'Length' column in descending order
	sorted_train_df = train_df_outlier.sort_values('num_stopwords', ascending=False)
	print_info(train_df_outlier)
	# print(''.join(train_df['article_text'][1]) )
	# print("////////"*3)
	for txt in sorted_train_df['article_text']:
		print( ''.join(txt[0:200]), '\n' )


def retrive_featdata():
	learner = Models()
	train_data = learner.model_import('../in_data/train_data-23jun.pkl')
	test_data = learner.model_import('../in_data/test_data-23jun.pkl')
	return train_data, test_data


def print_info(df):
	# Print the column names
	print("Column names:")
	print(df.columns)
	print()
	# Print the number of rows and columns
	print("Shape of the dataframe:")
	print(df.shape)
	print()
	# Print the summary statistics
	print("Summary statistics:")
	print(df.describe())
	print()
	# Print the first few rows
	print("First few rows:")
	print(df.head())
	print()


if __name__ == '__main__':
	test_simple()