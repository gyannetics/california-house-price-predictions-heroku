import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
# pd.set_option("display.float_format", .2f)
# np.set_printoptions(precision=9, suppress=True)
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# df = pd.concat([X,y], axis=1)
# print(df.head())
# X = X.values; y= y.values

# Preparing the model for training

X_train, X_test, y_train, y_test = train_test_split(X,y)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# print(train, test)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train) #, y_train)
test_scaled = scaler.transform(X_test)

# X_train_scaled = train_scaled[:,:-1]
# y_train_scaled = train_scaled[:,-1]
# X_test_scaled = test_scaled[:,:-1]
# y_test_scaled = test_scaled[:,-1]



# print(X_train.shape, X_test.shape)

# print (df.describe())



# Model Building

model = XGBRegressor(n_estimators=250,max_depth=6,) # criterion='absolute_error')

# Train the model
# model.fit(X_train, y_train)
model.fit(train_scaled, y_train)

#  Evaluate the model
y_pred = model.predict(test_scaled)

# y_pred = scaler.inverse_transform(y_pred_scaled)
# print(y_pred[:5])
# Scoring the model
score_train = model.score(train_scaled, y_train)
score_test = model.score(test_scaled, y_test)

# if score_train-score_test < 0.05:
with open("C:\\Users\\cogni\\Downloads\\iGyan\\heroku-ml-deploy\\saved_model\\model.pkl", 'wb') as f:
    joblib.dump(model, f)

error = mean_squared_error(y_true=y_test, y_pred=y_pred)
print(score_train, '\n', score_test, '\n', error)
