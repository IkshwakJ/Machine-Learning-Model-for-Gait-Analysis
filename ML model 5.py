import pandas as pd
import matplotlib
moment_data = pd.read_csv("Moment_data Test 2.csv")
print(moment_data.describe)
response = 'knee_angle_r_moment'
y = moment_data[[response]]
print(y)
predictors = list(moment_data.columns)
predictors.remove(response)
predictors.remove('time')
predictors.remove('ankle_angle_r_moment')
predictors.remove('knee_angle_l_moment')
predictors.remove('ankle_angle_l_moment')
x = moment_data[predictors]
print(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1234) 
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train, y_train)
print("Model Intercept: ", model.intercept_)
print("Coeff Values: ", model.coef_)
print("Score: ", model.score(x_test,y_test))
y_pred = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
print("Mean Squared Error", mean_absolute_error(y_test, y_pred))