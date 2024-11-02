import unittest
import pandas as pd
import numpy as np
from fase2 import evaluate_and_log_model, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys

# Ruta del archivo de datos
data_path = "forestfires_processed.csv"

class TestFase3(unittest.TestCase):
    # Prueba unitaria para verificar la carga correcta del archivo de datos
    def test_data_loading(self):
        """
        Verifica que el archivo de datos se cargue correctamente como un DataFrame de pandas y que no esté vacío.
        Escribe el resultado de la prueba en el archivo de salida.
        """
        data = pd.read_csv(data_path)
        resultado = "Pasó" if isinstance(data, pd.DataFrame) and not data.empty else "Falló"
        with open("resultados_pruebas.txt", "a", encoding="utf-8") as f:
            f.write("Prueba test_data_loading:\n")
            f.write(" - Objetivo: Verificar que el archivo de datos se cargue correctamente y no esté vacío.\n")
            f.write(f" - Resultado: {resultado}. Es un DataFrame no vacío.\n\n")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    # Prueba de integridad de datos: verifica si hay valores nulos en el dataset
    def test_data_integrity(self):
        """
        Verifica que no existan valores nulos en el dataset.
        Documenta si el dataset contiene o no valores nulos.
        """
        data = pd.read_csv(data_path)
        resultado = "Pasó" if not data.isnull().values.any() else "Falló"
        with open("resultados_pruebas.txt", "a", encoding="utf-8") as f:
            f.write("Prueba test_data_integrity:\n")
            f.write(" - Objetivo: Confirmar que el dataset no contiene valores nulos.\n")
            f.write(f" - Resultado: {resultado}. {'Sin valores nulos' if resultado == 'Pasó' else 'Contiene valores nulos'}.\n\n")
        self.assertFalse(data.isnull().values.any())

    # Prueba unitaria para la aplicación de One-Hot Encoding en variables categóricas
    def test_one_hot_encoding(self):
        """
        Comprueba que el One-Hot Encoding se aplica correctamente a las columnas categóricas 'month' y 'day'.
        Verifica la creación de columnas específicas y escribe los resultados.
        """
        data = pd.read_csv(data_path)
        X = data.drop(["area", "log_area"], axis=1)
        X_encoded = pd.get_dummies(X, columns=['month', 'day'])
        resultado = "Pasó" if "month_aug" in X_encoded.columns and "day_sun" in X_encoded.columns else "Falló"
        with open("resultados_pruebas.txt", "a", encoding="utf-8") as f:
            f.write("Prueba test_one_hot_encoding:\n")
            f.write(" - Objetivo: Validar que el One-Hot Encoding se aplica correctamente a las variables categóricas.\n")
            f.write(f" - Resultado: {resultado}. {'Columnas generadas correctamente' if resultado == 'Pasó' else 'Faltan columnas'}.\n\n")
        self.assertIn("month_aug", X_encoded.columns)
        self.assertIn("day_sun", X_encoded.columns)

    # Prueba para verificar la correcta división de los datos en entrenamiento y prueba
    def test_train_test_split(self):
        """
        Verifica que los datos se dividan correctamente en conjuntos de entrenamiento y prueba.
        Comprueba que el tamaño de los datos después de dividirlos coincida con el original.
        """
        data = pd.read_csv(data_path)
        X = data.drop(["area", "log_area"], axis=1)
        y = data["log_area"]
        X_encoded = pd.get_dummies(X, columns=['month', 'day'])
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        resultado = "Pasó" if len(X_train) + len(X_test) == len(X_encoded) and len(y_train) + len(y_test) == len(y) else "Falló"
        with open("resultados_pruebas.txt", "a", encoding="utf-8") as f:
            f.write("Prueba test_train_test_split:\n")
            f.write(" - Objetivo: Comprobar que los datos se dividen correctamente en entrenamiento y prueba.\n")
            f.write(f" - Resultado: {resultado}. {'División correcta' if resultado == 'Pasó' else 'División incorrecta'}.\n\n")
        self.assertEqual(len(X_train) + len(X_test), len(X_encoded))
        self.assertEqual(len(y_train) + len(y_test), len(y))

    # Prueba unitaria para evaluar y registrar todos los modelos de Machine Learning
    def test_evaluate_and_log_model(self):
        """
        Evalúa cada modelo en el diccionario 'models' y registra sus métricas RMSE y R².
        Documenta si cada modelo pasa los umbrales esperados.
        """
        data = pd.read_csv(data_path)
        X = data.drop(["area", "log_area"], axis=1)
        y = data["log_area"]
        X_encoded = pd.get_dummies(X, columns=['month', 'day'])
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        with open("resultados_pruebas.txt", "a", encoding="utf-8") as f:
            f.write("Prueba test_evaluate_and_log_model:\n")
            f.write(" - Objetivo: Evaluar y registrar cada modelo con precisión aceptable en RMSE y formato float en R².\n")
            
            for model_name, model in models.items():
                evaluate_and_log_model(model_name, model, X_train, X_test, y_train, y_test)
                y_pred = model.predict(X_test)

                # Calcular métricas
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                resultado_rmse = "Pasó" if rmse < 2.0 else "Falló"
                resultado_r2 = "Pasó" if isinstance(r2, float) else "Falló"
                
                f.write(f"   - {model_name}:\n")
                f.write(f"       - Resultado RMSE: {resultado_rmse}. Valor RMSE={rmse}\n")
                f.write(f"       - Resultado R²: {resultado_r2}. Valor R²={r2}\n\n")

                self.assertIsInstance(rmse, float)
                self.assertIsInstance(r2, float)
                self.assertLess(rmse, 2.0)

    # Prueba integral que valida el flujo completo de entrenamiento y evaluación para todos los modelos
    def test_complete_model_pipeline(self):
        """
        Ejecuta el flujo de entrenamiento y evaluación completo para cada modelo en el diccionario 'models'.
        Documenta el éxito o fracaso del proceso completo.
        """
        data = pd.read_csv(data_path)
        X = data.drop(["area", "log_area"], axis=1)
        y = data["log_area"]
        X_encoded = pd.get_dummies(X, columns=['month', 'day'])
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        errores = 0
        for model_name, model in models.items():
            try:
                evaluate_and_log_model(model_name, model, X_train, X_test, y_train, y_test)
            except Exception:
                errores += 1
        resultado = "Pasó" if errores == 0 else f"Falló con {errores} errores"
        with open("resultados_pruebas.txt", "a", encoding="utf-8") as f:
            f.write("Prueba test_complete_model_pipeline:\n")
            f.write(" - Objetivo: Validar que el flujo de entrenamiento y evaluación se ejecute correctamente para cada modelo.\n")
            f.write(f" - Resultado: {resultado}.\n\n")
        self.assertEqual(errores, 0)

# Configuración para redirigir la salida de las pruebas a un archivo
if __name__ == '__main__':
    with open("resultados_pruebas.txt", "a", encoding="utf-8") as f:
        # Ejecutar las pruebas y capturar los resultados en el archivo
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestFase3))
        
        # Escribir un resumen de los resultados al final del archivo
        f.write("\nResumen de Pruebas:\n")
        f.write(f"Total de pruebas ejecutadas: {result.testsRun}\n")
        f.write(f"Errores: {len(result.errors)}\n")
        f.write(f"Fallos: {len(result.failures)}\n")
        if result.wasSuccessful():
            f.write("Todas las pruebas pasaron exitosamente.\n")
        else:
            f.write("Algunas pruebas no pasaron.\n")