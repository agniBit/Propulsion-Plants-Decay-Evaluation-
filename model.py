import pickle
import os
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

if os.path.exists('propulsion.csv'):
	data = pd.read_csv(r'propulsion.csv')
else:
	raise 'propulsion.csv not exists'

model_name = 'model.pkl'
if os.path.exists(model_name):
	with open(model_name,'rb') as f: 
	    model, scaler = pickle.load(f)
else:
	raise 'model file not exists'

data.drop(['Unnamed: 0'],axis=1,inplace=True)

Y_c = data['GT Compressor decay state coefficient.']
Y_t = data['GT Turbine decay state coefficient.']

data.drop(['GT Compressor decay state coefficient.', 'GT Turbine decay state coefficient.'],axis=1,inplace=True)
X = data
del data

X = scaler.fit_transform(X)

y_pred = model.predict(X)

mse = mean_squared_error(Y_c, model.predict(X))
mae = mean_absolute_error(Y_c, model.predict(X))

print("mse = ",mse,"  mae = ",mae,"  rmse = ", math.sqrt(mse))