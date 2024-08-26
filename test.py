import pandas as pd
from joblib import load
import numpy as np

# Cargar el modelo y el scaler desde los archivos
with open('modelo_gradient_boosting.joblib', 'rb') as file:
    modelo_cargado = load.load(file)

with open('scaler.joblib', 'rb') as file:
    scaler_cargado = load.load(file)

# Cargar el nuevo conjunto de datos
nuevo_df = pd.read_csv('supply_chain_test.csv')

# Eliminar las columnas 'test_idx' y 'CLIENTNUM' del conjunto de prueba
nuevo_df.drop(columns=['test_idx', 'CLIENTNUM'], inplace=True)

# Realizar el mismo preprocesamiento que hiciste en el conjunto de entrenamiento
nuevo_df.replace('Unknown', np.nan, inplace=True)
nuevo_df = pd.get_dummies(nuevo_df, columns=["Gender", "Education_Level", 
                                             "Marital_Status", "Income_Category",
                                             "Card_Category"], drop_first=True)

# Seleccionar las columnas que se desean normalizar
columns_to_normalize = ['Customer_Age', 'Dependent_count', 'Months_on_book', 
                        'Total_Relationship_Count', 'Months_Inactive_12_mon',
                        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

# Normalizar las columnas correspondientes usando el scaler cargado
nuevo_df[columns_to_normalize] = scaler_cargado.transform(nuevo_df[columns_to_normalize])

# Asegurarse de que las columnas de nuevo_df coincidan con las del modelo entrenado
X_train_columns = modelo_cargado.best_estimator_.feature_names_in_

# Si faltan columnas en el conjunto de prueba, agregarlas con valores de 0
for col in X_train_columns:
    if col not in nuevo_df.columns:
        nuevo_df[col] = 0

# Reordenar las columnas en el mismo orden que X_train_columns
nuevo_df = nuevo_df[X_train_columns]

# Hacer las predicciones
predicciones = modelo_cargado.predict(nuevo_df)

# Mostrar las predicciones
print(predicciones)