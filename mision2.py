# Esto podría generar una advertencia en algunas versiones
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos de diabetes
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)
# Predecir y calcular el error cuadrático medio
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Error cuadrático medio:', mse)
