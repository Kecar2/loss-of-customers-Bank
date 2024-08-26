import pandas as pd
from joblib import dump
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Cargar el conjunto de datos de entrenamiento
df = pd.read_csv('supply_chain_train.csv')

# Hacer una copia del DataFrame original
df_train = df.copy()

# Reemplazar valores desconocidos con NaN
df_train.replace('Unknown', np.nan, inplace=True)

# Eliminar la columna 'train_idx', ya que no es necesaria para el modelo
df_train.drop(columns=['train_idx', 'CLIENTNUM'], inplace=True)

# Crear variables dummy para las columnas categóricas
df_train = pd.get_dummies(df_train, columns=["Gender", "Education_Level", 
                                             "Marital_Status", "Income_Category",
                                             "Card_Category"], drop_first=True)

# Normalización de datos
columns_to_normalize = ['Customer_Age', 'Dependent_count', 'Months_on_book', 
                        'Total_Relationship_Count', 'Months_Inactive_12_mon',
                        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

# Crear una instancia de MinMaxScaler y ajustar y transformar las columnas seleccionadas
scaler = MinMaxScaler()
df_train[columns_to_normalize] = scaler.fit_transform(df_train[columns_to_normalize])

# Definir la variable objetivo y las características
target = 'Attrition_Flag'
X = df_train.drop(target, axis=1)
y = df_train[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Definir un baseline (opcional)
baseline = y_train.mean()

# Configuración del modelo Gradient Boosting con GridSearchCV
gb = GridSearchCV(
    estimator=GradientBoostingClassifier(),
    param_grid={
        "max_depth": [4, 6, 8, 10],
        "min_samples_split": [20, 50, 70, 100],
    },
    cv=5,
    verbose=1,
    scoring="neg_mean_squared_error",
    return_train_score=True
)

# Entrenar el modelo
gb.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
gb_test_score = accuracy_score(
    y_pred=gb.predict(X_test),
    y_true=y_test
)

# Guardar el modelo entrenado y el scaler ajustado
with open('modelo_gradient_boosting.joblib', 'wb') as file:
    dump(gb, file)

with open('scaler.joblib', 'wb') as file:
    dump(scaler, file)

# Mostrar el puntaje de precisión del modelo
print(f'The test score with gradient boosting is {gb_test_score.round(3)}')