import pandas
import numpy


og_dataset = pandas.read_pickle("/home/christinejyeon/newspop/og_dataset_vec.pkl")
og_dataset = og_dataset.drop(og_dataset[og_dataset["vec"].isnull()].index)

feat, y = og_dataset["vec"].values, og_dataset[" shares"].values
frames = [pandas.DataFrame(og_dataset["vec"].values[i]) for i in range(og_dataset.shape[0])]
X = pandas.concat(frames)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=5)


from sklearn.preprocessing import StandardScaler
x_standardizer = StandardScaler()
y_standardizer = StandardScaler()
X_std = x_standardizer.fit_transform(X_train)
y_std = y_standardizer.fit_transform(y_train.reshape(-1,1))


from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.layers import Flatten
from keras.models import Model

model = Sequential()
#model.add(Flatten(input_shape=(1,512)))
model.add(Dense(150, activation='relu',input_shape=(512,)))
model.add(Dropout(0.5))
model.add(Dense(75, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
print(model.summary())
model.compile(loss='mse', optimizer='adam')

model.fit(X_std, y_std, epochs=30)

X_test_std = x_standardizer.fit_transform(X_test)
y_test_std = y_standardizer.fit_transform(y_test.reshape(-1,1))
y_test_pred = model.predict(X_test_std)

y_test_pred = numpy.squeeze(model.predict(X_test_std))
from scipy.stats.stats import pearsonr
pearsonr(y_test_pred, y_test_std)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_std, y_test_pred)
print(mse)

