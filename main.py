from load import Loader
from utils import Utils
from featuresHelper import FeaturesHelper
from textTransformer import Transformer
from models import Models


def test_model():
	loader = Loader()
	featHelper = FeaturesHelper()
	model = Models()

	# clasfr = model.model_import('./models/SVC_model_0.8342.pkl')
	# clasfr = model.model_import('./models/SVC_model_0.8169.pkl')
	clasfr = model.model_import('./models/VOTING_model_0.7608.pkl')
	print(clasfr.get_params())
	test = loader.load_from_csv('./in_data/test_data.csv')
	test_data = featHelper.add_features(test['article_text'])
	y_pred = clasfr.predict(test_data)
	# model.plot_own_confusion_matrix(test['Category'], y_pred)
	model.plot_roc(test['Category'], y_pred)

def learning_pipeline():
	print('Nuestro flujo de ML')

	loader = Loader()
	util = Utils()
	featHelper = FeaturesHelper()
	# transformer = Transformer()
	learner = Models()

	# data = loader.load_from_xlsx('./in_data/train.xlsx')
	# data = transformer.prepare_data(data)
	# data = loader.load_from_csv('./in_data/train_data.csv')
	# train, test = util.traint_test(data)
	train = loader.load_from_csv('./in_data/train_data.csv')
	test = loader.load_from_csv('./in_data/test_data.csv')
	# loader.save_to_csv(train, './in_data/train_data.csv')
	# loader.save_to_csv(test, './in_data/test_data.csv')
	# X, y = util.features_target(data, ['Link'], ['Category'])
	# print(X.info())
	# print(y.info())
	print('Datos de Entrenamiento:')
	print('Dataset class counts: ', util.get_class_counts(train))
	print('Dataset class proportions: ', util.get_class_proportions(train))
	print('Datos de prueba:')
	print('Dataset class counts: ', util.get_class_counts(test))
	print('Dataset class proportions: ', util.get_class_proportions(test))
	# print (test['Category'])
	train_data = featHelper.add_features(train['article_text'])
	test_data = featHelper.add_features(test['article_text'])

	# learner.model_export(train_data, './in_data/train_data.pkl')
	# learner.model_export(test_data, './in_data/test_data.pkl')
	# train_data = learner.model_import('./in_data/train_data.pkl')
	# test_data = learner.model_import('./in_data/test_data.pkl')

	# print(train_data.info())

	# featHelper.plot_distr_cols(train_data)
	# featHelper.plot_distr_corr(train_data, train['Category'])
	# featHelper.plot_corr_matrix(train_data, 8)

	# print( train_data['article_text'])
	# print(train_data.columns)
	# train_data_cate = train['Category']
	# train_data.drop(columns=['Category'], inplace=True)
	learner.pipeline_learning(train_data, train['Category'], test_data, test['Category'])

	# model.grid_training(X, y)


def generate_data_features():
	loader = Loader()
	util = Utils()
	featHelper = FeaturesHelper()
	learner = Models()

	train = loader.load_from_csv('./in_data/train_data.csv')
	test = loader.load_from_csv('./in_data/test_data.csv')
	# loader.save_to_csv(train, './in_data/train_data.csv')
	# loader.save_to_csv(test, './in_data/test_data.csv')
	# X, y = util.features_target(data, ['Link'], ['Category'])

	print('Datos de Entrenamiento:')
	print('Dataset class counts: ', util.get_class_counts(train))
	print('Dataset class proportions: ', util.get_class_proportions(train))
	print('Datos de prueba:')
	print('Dataset class counts: ', util.get_class_counts(test))
	print('Dataset class proportions: ', util.get_class_proportions(test))
	# print (test['Category'])
	train_data = featHelper.add_features(train['article_text'])
	test_data = featHelper.add_features(test['article_text'])

	learner.model_export(train_data, './in_data/train_data-23jun.pkl')
	learner.model_export(test_data, './in_data/test_data-23jun.pkl')
	# train_data = learner.model_import('./in_data/train_data.pkl')
	# test_data = learner.model_import('./in_data/test_data.pkl')


if __name__ == '__main__':
	generate_data_features()
	# learning_pipeline()
	# test_model()
