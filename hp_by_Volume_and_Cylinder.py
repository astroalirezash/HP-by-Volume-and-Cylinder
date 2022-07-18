import pandas as pn
from sklearn import linear_model

df = pn.read_csv('cars.csv')

X = df[['Volume', 'Cylinder']]
y = df['hp']

regr = linear_model.LinearRegression()
regr.fit(X, y)

v = float(input('Insert Volume of the engine: '))
c = float(input('Insert number of Cylinders: '))

predicthp = regr.predict([[v, c]])

print(predicthp[0])
