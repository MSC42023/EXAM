import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset

dataframe = pd.read_csv("california_housing_train.csv",delim_whitespace=True, header=None)
dataset=dataframe.values
X =dataset[:, 0:13]
Y =dataset[:, 1:13]

#define Accuracy model 
def accuracy_model():
  #create model
  model_Sequential()
  model.add(Dense(15, input_dim-13,
                  kernel_initializer='normal', activation="relu")) 
  model.add(Dense(13, kernel_initializer='normal', activation="relu")) 
  model.add(Dense(1, kernel_initializer='normal'))
  # Compile model
  model.compile(loss="mean_squared_error", optimizer="adan")
  return model
  estimators = []
  estimators.append(('standardize', StandardScaler())) 
  estimators.append(('mlp', KerasRegressor(
      build_fn=accuracy_model, epochs=10, batch_size=5)))
  pipeline = Pipeline (estimators)
  kfold = KFold(n_splits-10)
  results - cross_val_score (pipeline, X, Y, cv=kfold)
  print("\n")
  print("Accuracy: %.2f (%.2f) MSE" (results.mean(), results.std()))
