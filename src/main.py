import pkg.GradientDescent as gd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/insurance.csv")
df.head()
train_df, test_df = train_test_split(df, test_size=0.8, random_state=123)
x_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
y_col = "charges"

x_train = train_df[x_cols]
y_train = train_df[y_col]

x_test = train_df[x_cols]
y_test = train_df[y_col]

model = gd.RegressionGD(x_train, y_train, 100000)
model.gradient_descent()
y_hat = model.predict(x_test)
mse = mean_squared_error(y_test, y_hat)
rmse = np.sqrt(mse)

print(f"RMSE of Linear Regression: {rmse:5}")
print('The model has been trained successfully!!')
while True:
    print("Please enter the following details:")
    age = int(input("Age: "))
    sex = input("Sex (male/female): ")
    bmi = int(input("BMI: "))
    children = int(input("Number of children: "))
    smoker = input("Smoker (yes/no): ")
    region = input("Region (northeast/northwest/southeast/southwest): ")

    user_data = pd.DataFrame([{
        "age": age, 
        "sex": sex, 
        "bmi": bmi, 
        "children": children, 
        "smoker": smoker, 
        "region": region
    }])

    prediction = model.predict(user_data)
    print(f"Predicted Insurance Charges: ${prediction[0]:.2f}")
    check = input("Do you want to continiue? (yes/no)")
    if check =='no':
        break