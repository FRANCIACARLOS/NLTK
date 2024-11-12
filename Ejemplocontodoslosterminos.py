# importa bibliotecas para realizar el procesamiento de datos
import numpy as np #manejo de datos númericos
from sklearn.pipeline import Pipeline # Permite combinar diferentes pasos de preprocesamiento
from sklearn.ensemble import RandomForestClassifier # Clasifica aleatoriamento el bosque de datos
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer #metodos de escalado
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

# Datos sintéticos con características numéricas y etiquetas para ilustrar el proceso de escalado y clasificación
X = np.array([
    [1, 2000, 50],
    [2, 3000, 70],
    [3, 1500, 40],
    [4, 3500, 80],
    [5, 2500, 60],
    [6, 1200, 30],
    [7, 4000, 90],
    [8, 2200, 55]
])
y = np.array([0, 1, 0, 1, 1, 0, 1, 0])

# División de los datos en conjuntos de entrenamiento y prueba (X_train y y_train son los datos de entrenamiento; X_test y y_test son los datos de prueba.)
# random_state=42 garantiza que la división sea la misma cada vez que se ejecuta el código
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 1. Feature Scaling usando StandardScaler (StandardScaler convierte los datos para que tengan media 0 y desviación estándar 1)
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

# Entrenamiento y evaluación con escalado estándar (StandardScaler)
clf_standard = RandomForestClassifier()
clf_standard.fit(X_train_standard, y_train)
y_pred_standard = clf_standard.predict(X_test_standard)
accuracy_standard = accuracy_score(y_test, y_pred_standard)
print("Accuracy con StandardScaler:", accuracy_standard)

# 2. Escalado de funciones usando MinMaxScaler (Normalización)
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# Entrenamiento y evaluación con normalización MinMaxScaler
clf_minmax = RandomForestClassifier()
clf_minmax.fit(X_train_minmax, y_train)
y_pred_minmax = clf_minmax.predict(X_test_minmax)
accuracy_minmax = accuracy_score(y_test, y_pred_minmax)
print("Accuracy con MinMaxScaler:", accuracy_minmax)

# 3. Escalado de características conRobustScaler
scaler_robust = RobustScaler()
X_train_robust = scaler_robust.fit_transform(X_train)
X_test_robust = scaler_robust.transform(X_test)

# Entrenamiento y evaluación con escalado robusto (RobustScaler)
clf_robust = RandomForestClassifier()
clf_robust.fit(X_train_robust, y_train)
y_pred_robust = clf_robust.predict(X_test_robust)
accuracy_robust = accuracy_score(y_test, y_pred_robust)
print("Accuracy con RobustScaler:", accuracy_robust)

# 4. Feature Scaling usando Normalizer (L2 Normalization)
scaler_normalizer = Normalizer()
X_train_normalized = scaler_normalizer.fit_transform(X_train)
X_test_normalized = scaler_normalizer.transform(X_test)

# Entrenamiento y evaluación con Normalizer
clf_normalizer = RandomForestClassifier()
clf_normalizer.fit(X_train_normalized, y_train)
y_pred_normalizer = clf_normalizer.predict(X_test_normalized)
accuracy_normalizer = accuracy_score(y_test, y_pred_normalizer)
print("Accuracy con Normalizer:", accuracy_normalizer)

# 5. Uso de Pipeline con StandardScaler y RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Cambia aquí el escalador para probar diferentes métodos
    ('classifier', RandomForestClassifier())
])

# Entrenamiento y evaluación usando pipeline
pipeline.fit(X_train, y_train)
pipeline_accuracy = pipeline.score(X_test, y_test)
print("Accuracy usando pipeline con StandardScaler:", pipeline_accuracy)
