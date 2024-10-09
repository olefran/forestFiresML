import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Ruta del dataset
file_path = 'C:/Users/luisd/Desktop/Posgrado IMA/MLOP/forestfires.csv'

# Cargar el dataset
data = pd.read_csv(file_path)

# Copiar los datos para evitar modificar el original
processed_data = data.copy()

# Codificar 'month' y 'day'
label_encoder_month = LabelEncoder()
label_encoder_day = LabelEncoder()
processed_data['month'] = label_encoder_month.fit_transform(processed_data['month'])
processed_data['day'] = label_encoder_day.fit_transform(processed_data['day'])

# Aplicar transformación logarítmica a 'area'
processed_data['area'] = np.log(processed_data['area'] + 1)

# Eliminar outliers en 'area'
Q1 = processed_data['area'].quantile(0.25)
Q3 = processed_data['area'].quantile(0.75)
IQR = Q3 - Q1
processed_data = processed_data[~((processed_data['area'] < (Q1 - 1.5 * IQR)) | 
                                  (processed_data['area'] > (Q3 + 1.5 * IQR)))]

# Estandarizar características numéricas
scaler = StandardScaler()
numerical_features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
processed_data[numerical_features] = scaler.fit_transform(processed_data[numerical_features])

# Dividir los datos en características y variable objetivo
X = processed_data.drop(columns=['area'])
y = processed_data['area']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar un diccionario para los resultados
results = {}

# Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
results["Regresión Lineal"] = {'RMSE': rmse_lr, 'R²': r2_lr}

# Árbol de Decisión
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
r2_tree = r2_score(y_test, y_pred_tree)
results["Árbol de Decisión"] = {'RMSE': rmse_tree, 'R²': r2_tree}

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
results["Random Forest"] = {'RMSE': rmse_rf, 'R²': r2_rf}

# SVM
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))
r2_svm = r2_score(y_test, y_pred_svm)
results["SVM"] = {'RMSE': rmse_svm, 'R²': r2_svm}

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)
results["Gradient Boosting"] = {'RMSE': rmse_gb, 'R²': r2_gb}

# Imprimir resultados
print("\nResultados de Evaluación:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  R²: {metrics['R²']}")
    print("\n")

# Visualización de resultados
model_names = list(results.keys())
rmse_values = [metrics['RMSE'] for metrics in results.values()]
plt.figure(figsize=(10, 6))
plt.barh(model_names, rmse_values, color='skyblue')
plt.xlabel('RMSE')
plt.title('Comparación de RMSE entre Modelos')
plt.show()

# Guardar los datos procesados en archivos CSV
X_train.to_csv('X_train_processed.csv', index=False)
X_test.to_csv('X_test_processed.csv', index=False)
y_train.to_csv('y_train_processed.csv', index=False)
y_test.to_csv('y_test_processed.csv', index=False)

# Guardar el dataset final procesado
processed_data.to_csv('forestfires_processed.csv', index=False)
print("\nLos datos han sido procesados y guardados en archivos CSV.")
