import yaml
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import mlflow

class TrainModels():
    def __init__(self, params_path):
        with open(params_path, 'r') as ymlfile:
            self.params = yaml.safe_load(ymlfile)

        self.model_dir = self.params['data']['models']

    def train_models(
            self, 
            X_train_path, y_train_path, 
            X_test_path, y_test_path, 
            mlserverURI="http://127.0.0.1:5000"
    ):
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)
        
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

                model_path = f"{self.model_dir}/{model_name}.pkl"
                joblib.dump(model, model_path)

if __name__ == '__main__':
    params_path = sys.argv[1]
    X_train_path = sys.argv[2]
    y_train_path = sys.argv[3]
    X_test_path = sys.argv[4]
    y_test_path = sys.argv[5]

    trm = TrainModels(params_path)
    trm.train_models(X_train_path, y_train_path, X_test_path, y_test_path)
