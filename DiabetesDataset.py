import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


class DiabetesDataset:

    def __init__(self, short_desc, filename, name_features, name_target):
        self.descr = short_desc
        self.file_name = filename
        self.features_name = list(name_features)
        self.target_names = list(name_target)
        df = pd.read_csv(filename)
        self.features = df.iloc[:, :len(self.features_name)]
        self.target = df.iloc[:, -len(self.target_names)]
        self.features_scaled = None

        self.X_train_val = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train_val = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def splitDataSetBasic(self, t_size, rnd_state, scaled_features: bool):
        if (scaled_features):
            self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.features_scaled,
                                                                                            self.target,
                                                                                            test_size=t_size,
                                                                                            random_state=rnd_state,
                                                                                            shuffle=True)
        else:
            self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.features,
                                                                                            self.target,
                                                                                            test_size=t_size,
                                                                                            random_state=rnd_state,
                                                                                            shuffle=True)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val,
                                                                              test_size=0.11, random_state=rnd_state,
                                                                              shuffle=True)
