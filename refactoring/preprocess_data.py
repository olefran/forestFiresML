import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class PreprocessData():
    def __init__(self, data_path):
        self.data_ = pd.read_csv(data_path)

    def preprocess_data(self):
        processed_data = self.data_.dropna()

        processed_data['area'] = np.log(processed_data['area'] + 1)
        processed_data['area'] = np.log(processed_data['area'] + 1)

        X = processed_data.drop(columns=['area'])
        y = processed_data['area']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        label_encoder_month = LabelEncoder()
        X_train['month'] = label_encoder_month.fit_transform(X_train['month'])
        X_test['month'] = label_encoder_month.transform(X_test['month'])

        label_encoder_day = LabelEncoder()
        X_train['day'] = label_encoder_day.fit_transform(X_train['day'])
        X_test['day'] = label_encoder_day.transform(X_test['day'])

        scaler = StandardScaler()
        numerical_features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.fit_transform(X_test[numerical_features])

        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    prd = PreprocessData(data_path)
    X_train, X_test, y_train, y_test = prd.preprocess_data()

    X_train.to_csv(output_train_features, index=False)
    X_test.to_csv(output_test_features, index=False)
    y_train.to_csv(output_train_target, index=False)
    y_test.to_csv(output_test_target, index=False)