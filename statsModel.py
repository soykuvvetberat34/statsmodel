import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#multiple linear regression
datas=pd.read_csv("Turkcell Makine Öğrenmesi//maas.csv")
x=datas.iloc[:,[0,2]].values
y=datas[["maas"]].values
print(x)
print(y)
model=sm.OLS(y,x).fit()
print(model.summary())
model_lin_reg=LinearRegression()
model_lin_reg.fit(x,y)
predict=model_lin_reg.intercept_ + model_lin_reg.coef_*x
predict=np.sqrt(np.mean(predict**2))
print(model_lin_reg.coef_)
print(model_lin_reg.intercept_)
print(predict)
#statsmodelin ve sklearn verdiği b0 ve b1 değerleri farklı cıkıyopr




















