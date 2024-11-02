#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os


# In[2]:


# Configuración de la URI de MLflow
mlflow.set_tracking_uri("file:///mnt/mlruns")  
mlflow.set_experiment("forest_fires_experiment")

# Carga de datos del archivo CSV proporcionado
data_path = "forestfires_processed.csv"  # Ruta al archivo CSV
data = pd.read_csv(data_path)

# Separar las características (X) y la variable objetivo (y)
X = data.drop(["area", "log_area"], axis=1)
y = data["log_area"]


# In[3]:


# Aplicar One-Hot Encoding a las columnas categóricas ('month', 'day')
X = pd.get_dummies(X, columns=['month', 'day'])

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


# Definir función para registrar y evaluar modelos
def evaluate_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Calcular RMSE (cambio recomendado)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Cambiar a root_mean_squared_error si tu versión de sklearn lo permite
        
        # Calcular R²
        r2 = r2_score(y_test, y_pred)
        
        # Registrar parámetros, métricas y el modelo
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Registrar el modelo con un ejemplo de entrada para inferir la firma del modelo
        input_example = X_test.iloc[0:1]  # Ejemplo de entrada para MLflow
        mlflow.sklearn.log_model(model, model_name, input_example=input_example)
        
        # Imprimir resultados
        print(f"{model_name}: RMSE={rmse}, R²={r2}")


# In[5]:


# Modelos a evaluar
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=4, random_state=42),
    'SVM': SVR(kernel='rbf'),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}


# In[6]:


# Evaluar cada modelo
for model_name, model in models.items():
    evaluate_and_log_model(model_name, model, X_train, X_test, y_train, y_test)


# In[7]:


# Guardar todos los modelos entrenados usando Joblib
for model_name, model in models.items():
    model_filename = f'models/{model_name.replace(" ", "_").lower()}.pkl'
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, model_filename)
    print(f"Modelo {model_name} guardado en {model_filename}")


# In[8]:


# Versionar todos los modelos con DVC
os.system("dvc add models/")
os.system("dvc push")


# In[9]:


# Versionar los datos procesados que usaste
os.system("dvc add forestfires_processed.csv")
os.system("dvc push")

