# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#



# flake8: noqa: E501
"""Tarea 1"""

import gzip
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataframe.

    Args:
        df (pd.DataFrame): dataframe a limpiar.

    Returns:
        pd.DataFrame: dataframe limpio.
    """
    print("Limpiando datos...")
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df


def create_pipeline() -> Pipeline:
    """Crea un pipeline para el modelo.

    Returns:
        Pipeline: pipeline para el modelo.
    """
    print("Creando pipeline...")
    pipeline = Pipeline(
        [
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )
    return pipeline

def optimize_hyperparameters(
    pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """Optimiza los hiperparámetros del pipeline.

    Args:
        pipeline (Pipeline): pipeline a optimizar.
        x_train (pd.DataFrame): datos de entrenamiento.
        y_train (pd.Series): etiquetas de entrenamiento.

    Returns:
        GridSearchCV: pipeline optimizado.
    """
    print("Optimizando hiperparámetros con validación cruzada (10 splits)...")
    param_grid = {
    "model__n_estimators": [75],  # Prueba más árboles
    "model__max_depth": [50],  # 20, 130 Diferentes profundidades
    "model__min_samples_split": [20],  # Diferentes tamaños de división
    "model__min_samples_leaf": [1],  # Diferentes tamaños de hoja
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=10, scoring="balanced_accuracy", verbose=2
    )  # verbose=2 para mostrar progreso
    grid_search.fit(x_train, y_train)
    
    print("Mejor conjunto de hiperparámetros encontrados:", grid_search.best_params_)
    
    return grid_search


def calculate_metrics(
    model: GridSearchCV, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
) -> list:
    """Calcula las métricas para el modelo.

    Args:
        model (GridSearchCV): modelo a evaluar.
        x_train (pd.DataFrame): datos de entrenamiento.
        y_train (pd.Series): etiquetas de entrenamiento.
        x_test (pd.DataFrame): datos de prueba.
        y_test (pd.Series): etiquetas de prueba.

    Returns:
        list: lista de métricas.
    """
    print("Calculando métricas...")
    metrics = []
    for dataset, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        y_pred = model.predict(x)
        metrics.append(
            {
                "type": "metrics",
                "dataset": dataset,
                "precision": precision_score(y, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1_score": f1_score(y, y_pred),
            }
        )
        cm = confusion_matrix(y, y_pred)
        metrics.append(
            {
                "type": "cm_matrix",
                "dataset": dataset,
                "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
                "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
            }
        )
    return metrics


def main():
    """Función principal."""

    print("Cargando datos...")
    train_df = pd.read_csv("files/input/train_data.csv.zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip")

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    print("Dividiendo datos...")
    x_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    x_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    pipeline = create_pipeline()

    model = optimize_hyperparameters(pipeline, x_train, y_train)

    print("Guardando modelo...")
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(model, file)

    metrics = calculate_metrics(model, x_train, y_train, x_test, y_test)

    print("Guardando métricas...")
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for metric in metrics:
            file.write(json.dumps(metric) + "\n")

    print("Proceso completado.")

    # Imprimir la precisión del modelo
    precision_train = next(
        metric["precision"] for metric in metrics if metric["dataset"] == "train"
    )
    precision_test = next(
        metric["precision"] for metric in metrics if metric["dataset"] == "test"
    )
    print(f"Precisión en entrenamiento: {precision_train:.4f}")
    print(f"Precisión en prueba: {precision_test:.4f}")

    print("Proceso completado.")


if __name__ == "__main__":
    main()