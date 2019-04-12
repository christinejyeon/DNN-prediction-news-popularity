import pandas
import numpy
import pickle
import cPickle as pickle
from matplotlib import pyplot


og_dataset = pandas.read_pickle("/Users/Christine/Documents/GLIS 689/og_dataset_vec.pkl")

og_dataset = pandas.read_pickle("/home/christinejyeon/newspop/og_dataset_vec.pkl")

og_dataset = pandas.read_pickle("/home/christinejyeon/newspop/og_dataset_vec.pkl")
og_dataset = og_dataset.drop(og_dataset[og_dataset["vec"].isnull()].index)
og_dataset = og_dataset.drop(og_dataset[og_dataset[" shares"].isnull()].index)

feat, y = og_dataset["vec"].values, og_dataset[" shares"].values
frames = [pandas.DataFrame(og_dataset["vec"].values[i]) for i in range(og_dataset.shape[0])]
X = pandas.concat(frames)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=5)

from sklearn.preprocessing import MinMaxScaler
x_standardizer = MinMaxScaler()
y_standardizer = MinMaxScaler()
X_std = x_standardizer.fit_transform(X_train)
y_std = y_standardizer.fit_transform(y_train.reshape(-1,1))


from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.layers import Flatten
from keras.models import Model

model = Sequential()
model.add(Dense(150, activation='relu',input_shape=(512,)))
model.add(Dropout(0.5))
model.add(Dense(75, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
print(model.summary())
model.compile(loss='mse', optimizer='adam')

X_val_std = x_standardizer.fit_transform(X_val)
y_val_std = y_standardizer.fit_transform(y_val.reshape(-1,1))
X_test_std = x_standardizer.fit_transform(X_test)
y_test_std = y_standardizer.fit_transform(y_test.reshape(-1,1))

temp_history = model.fit(X_std, y_std, epochs=50, batch_size=32, validation_data=(X_val_std, y_val_std))

pyplot.plot(temp_history.history['loss'])
pyplot.plot(temp_history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
pyplot.clf()
#pyplot.savefig("train_validation_loss_1.png")

y_test_pred = model.predict(X_test_std)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_std, y_test_pred)
print(mse)
# 0.0003099916134197465 with optimizer adam, batch_size 128
# 0.00030457022894766 with optimizer adam, batch_size 32

# split the data into 3 datasets, mse got bigger. 0.00041118366644753444
# the validation loss became higher than usual too , around 0.0008


from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
seed = 7
numpy.random.seed(seed)
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(150, activation='relu',input_shape=(512,)))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(75, activation='relu'))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(1, activation='linear'))
    regressor.compile(loss='mse', optimizer='adam')
    return regressor
from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn = build_regressor, batch_size = 32, epochs = 30)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(regressor, X_std, y_std, cv=kfold)
print("Results: %.5f (%.5f) MSE" % (results.mean(), results.std()))


def build_larger_regressor():
    regressor = Sequential()
    regressor.add(Dense(256, activation='relu',input_shape=(512,)))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(150, activation='relu'))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(75, activation='relu'))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(1, activation='linear'))
    regressor.compile(loss='mse', optimizer='adam')
    return regressor
regressor = KerasRegressor(build_fn = build_larger_regressor, batch_size = 32, epochs = 30)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(regressor, X_std, y_std, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))



accuracies = cross_val_score(estimator = regressor, X = X_std, y = y_std,scoring='r2',cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()










################################### Added other columns ###################################

notog_X = X.copy()
temp = og_dataset.drop(["0","vec"," shares"], axis=1)
notog_X = pandas.concat([notog_X.reset_index(drop=True), temp.reset_index(drop=True)], axis=1)
X_train, X_test, y_train, y_test = train_test_split(notog_X,y,test_size=0.3,random_state=5)
X_std = x_standardizer.fit_transform(X_train)
y_std = y_standardizer.fit_transform(y_train.reshape(-1,1))

model = Sequential()
model.add(Dense(150, activation='relu',input_shape=(571,)))
model.add(Dropout(0.5))
model.add(Dense(75, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
print(model.summary())
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

X_test_std = x_standardizer.fit_transform(X_test)
y_test_std = y_standardizer.fit_transform(y_test.reshape(-1,1))

temp_history = model.fit(X_std, y_std, epochs=30, batch_size=32, validation_data=(X_test_std, y_test_std))

pyplot.plot(temp_history.history['loss'])
pyplot.plot(temp_history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
pyplot.clf()
#pyplot.savefig("train_validation_loss_1.png")

y_test_pred = model.predict(X_test_std)
mse = mean_squared_error(y_test_std, y_test_pred)
print(mse)
#0.00030380693708801954

