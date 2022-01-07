"""     AQUI CARGAREMOS LOS DATOS QUE SE OBTENGAN DE LSO DATOS      """
import pandas as pd


class Loader:
	filename = r'/media/miguel_morales/DATA/UPIITA/10mo_SEMESTRE/PT_II/Clasificador/Corpus/train.xlsx'
	filename_test = r'/media/miguel_morales/DATA/UPIITA/10mo_SEMESTRE/PT_II/Clasificador/Corpus/development.xlsx'

	def load_from_csv(self, path):
		return pd.read_csv(path, sep=';')

	def load_from_xlsx(self, path):
		df = pd.read_excel(path, engine='openpyxl')
		# num = 0
		# for i in df['index']:
		#     if i != num:
		#         print(num, i, sep=', ', end='\n')
		#         num = i
		#     num = num + 1

		return df

	def load_from_posgressql(self):
		pass

	def save_to_csv(self, df, path):
		df.to_csv(path, index=False, sep=';')

	def append_to_csv(self,df, path):
		df.to_csv(path, index=False, sep=';', mode='a', header=False)
