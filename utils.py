import pandas as pd
from sklearn.model_selection import train_test_split


class Utils:

    def features_target(self, dataset, drop_cols, y_col):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y_col]
        return X, y

    def get_class_counts(self, df):
        grp = df.groupby(['Category'])['subtitle'].nunique()
        return {key: grp[key] for key in list(grp.keys())}

    def get_class_proportions(self, df):
        class_counts = self.get_class_counts(df)
        return {val[0]: round(val[1]/df.shape[0], 4) for val in class_counts.items()}

     # TODO: Obtener una mejor reparticion de los datos a traves de stratify
    def traint_test(self, df):
        train, test = train_test_split(df, test_size=0.2, stratify=df['Category'])
        map_proportions = self.get_class_proportions(train)
        delta, count = 0.009, 1
        while abs(map_proportions[0] - 0.5327) > delta:
            train, test = train_test_split(df, test_size=0.2, stratify=df['Category'])
            map_proportions = self.get_class_proportions(train)
            count += 1
        # Tenemos una buena separacion de los datos...
        print( count )
        return train, test
