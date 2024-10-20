# Setup

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import mlflow
from mlflow.models import infer_signature


#Defining dataset
class DataImport:
    @staticmethod
    def data_fetch():
        # 162 for forest fires
        dataset_ = fetch_ucirepo(id=162)
        X = dataset_.data.features
        y = dataset_.data.targets
        return pd.concat([X, y], axis = 1)

# Creating the classes
class DataExplorer:
    @staticmethod
    def explore_data(data_):
        print(data_.head().T)
        print(data_.describe())
        print(data_.info())

    @staticmethod
    def plot_histograms(data_):
        data_.hist(bins=15, figsize=(15, 10))
        plt.show()

    def data_exp_n_prep(data_):
        print("\nValores nulos por columna:")
        print(data_.isnull().sum())

class FF_model:
    def __init__(self, data_):
        self.data_ = data_

    def data_prep(self):

        data = self.data_.dropna()

        print(f"\nNúmero de filas duplicadas: {data.duplicated().sum()}")
        data = data.drop_duplicates()
        print(f"Número de filas después de eliminar duplicados: {data.shape[0]}")

        Q1 = data['area'].quantile(0.25)
        Q3 = data['area'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data_filtered = data[(data['area'] >= lower_bound) & (data['area'] <= upper_bound)]
        print(f"Cantidad de datos después de eliminar outliers: {data_filtered.shape[0]}")

        data_filtered['log_area'] = np.log1p(data_filtered['area'])

        data_filtered.to_csv(f'../data/processed/{"forest_fires_prepared_df.csv"}', index=False)

    def data_processing(self):

        processed_data = pd.read_csv(f'./data/processed/{"forest_fires_prepared_df.csv"}')

        label_encoder_month = LabelEncoder()
        label_encoder_day = LabelEncoder()
        processed_data['month'] = label_encoder_month.fit_transform(processed_data['month'])
        processed_data['day'] = label_encoder_day.fit_transform(processed_data['day'])

        processed_data['area'] = np.log(processed_data['area'] + 1)

        scaler = StandardScaler()
        numerical_features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
        processed_data[numerical_features] = scaler.fit_transform(processed_data[numerical_features])

        X = processed_data.drop(columns=['area'])
        y = processed_data['area']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        file_path = f'./data/processed'

        X_train.to_csv(f'{file_path}/X_train_processed.csv', index=False)
        X_test.to_csv(f'{file_path}/X_test_processed.csv', index=False)
        y_train.to_csv(f'{file_path}/y_train_processed.csv', index=False)
        y_test.to_csv(f'{file_path}/y_test_processed.csv', index=False)

        processed_data.to_csv(f'{file_path}/forest_fires_processed_df.csv', index=False)

    def model_creation(self,file_path=f'./data/processed'):

        X_train = pd.read_csv(f'{file_path}/X_train_processed.csv')
        X_test = pd.read_csv(f'{file_path}/X_test_processed.csv')
        y_train = pd.read_csv(f'{file_path}/y_train_processed.csv')
        y_test = pd.read_csv(f'{file_path}/y_test_processed.csv')

        results = {}

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred_lr = linear_model.predict(X_test)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        r2_lr = r2_score(y_test, y_pred_lr)
        results["Regresión Lineal"] = {'RMSE': rmse_lr, 'R²': r2_lr}

        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train['area'])
        y_pred_rf = rf_model.predict(X_test)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        r2_rf = r2_score(y_test, y_pred_rf)
        results["Random Forest"] = {'RMSE': rmse_rf, 'R²': r2_rf}

        svm_model = SVR(kernel='rbf')
        svm_model.fit(X_train, y_train['area'])
        y_pred_svm = svm_model.predict(X_test)
        rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))
        r2_svm = r2_score(y_test, y_pred_svm)
        results["SVM"] = {'RMSE': rmse_svm, 'R²': r2_svm}

        gb_model = GradientBoostingRegressor(random_state=42)
        gb_model.fit(X_train, y_train['area'])
        y_pred_gb = gb_model.predict(X_test)
        rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
        r2_gb = r2_score(y_test, y_pred_gb)
        results["Gradient Boosting"] = {'RMSE': rmse_gb, 'R²': r2_gb}

        print("\nResultados de Evaluación:")
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            print(f"  RMSE: {metrics['RMSE']}")
            print(f"  R²: {metrics['R²']}")
            print("\n")

        model_names = list(results.keys())
        rmse_values = [metrics['RMSE'] for metrics in results.values()]
        plt.figure(figsize=(10, 6))
        plt.barh(model_names, rmse_values, color='skyblue')
        plt.xlabel('RMSE')
        plt.title('Comparación de RMSE entre Modelos, corrida de prueba')
        plt.show()


    def mlflow_tracking(self,file_path=f'./data/processed',mlserverURI="http://127.0.0.1:5000"):

        X_train = pd.read_csv(f'{file_path}/X_train_processed.csv')
        X_test = pd.read_csv(f'{file_path}/X_test_processed.csv')
        y_train = pd.read_csv(f'{file_path}/y_train_processed.csv')
        y_test = pd.read_csv(f'{file_path}/y_test_processed.csv')

        #MLFflow Tracking
        mlflow.set_tracking_uri(mlserverURI)
        mlflow.set_experiment("MLForestFires")

        models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "SVM": SVR(kernel='rbf'),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
        }

        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                # Calculate R2
                r2 = r2_score(y_test, y_pred)
                 # Calculate MAD
                mad = mean_absolute_error(y_test, y_pred)

                mlflow.log_param("model", model_name)

                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mad", mad)

                mlflow.sklearn.log_model(model, model_name)

                mlflow.end_run()

    def best_model_run(self):
        # Fetch all runs from the experiment
        experiment_id = mlflow.get_experiment_by_name("MLForestFires").experiment_id
        runs = mlflow.search_runs(experiment_ids=[experiment_id])

        # Find the run with the lowest RMSE
        best_run = runs.sort_values("metrics.rmse", ascending=True).iloc[0]

        print(f"Best run is the model {best_run['tags.mlflow.runName']} with RMSE: {best_run['metrics.rmse']} with ID:{best_run['run_id']}")
