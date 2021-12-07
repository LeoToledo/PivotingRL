import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from keras.models import load_model
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
import IPython
import warnings
warnings.filterwarnings("ignore")

class DataImputer:
    def __init__(self, path_to_data=os.getcwd()):
        # Read the data
        self.df = pd.read_csv(f'{path_to_data}/logs/logs.csv').sample(frac=0.1)

        self.model = None
        self.trainX_df, self.trainY_df = None, None
        self.testX_df, self.testY_df = None, None
        self.predictions = None

    def split_train_test(self, test_size):

        # 'force_0','force_1','force_2','force_3','force_4','force_5','force_6','force_7','force_8'
        self.trainX_df, self.testX_df, \
        self.trainY_df, self.testY_df = train_test_split(self.df[['force_0','force_1','force_2','force_3','force_4','force_5','force_6','force_7','force_8']],
                                                         self.df['tools_angle'], test_size=test_size, random_state=42)
        # IPython.embed()

    def process_data(self, test_size):
        # Calculates the tool angle with relation to the gripper
        # self.df['tools_angle'] = self.df['tools_angle'] - self.df['grippers_angle']

        # Splits train and test data
        self.split_train_test(test_size=test_size)

    def test_model_locally(self, testX=None, path_to_model=os.getcwd()):
        if testX is None:
            testX = self.testX_df

        model = load_model(f'models/{path_to_model}/simple_mlp_model.hdf5')

        self.predictions = model.predict(testX)

        return self.predictions

    def test_model(self, model=None,testX=None):
        if testX is None:
            testX = self.testX_df

        if model is None:
            model = load_model(f'{os.getcwd()}/utils/models/saved_models/simple_mlp_model.hdf5')

        self.predictions = model.predict(testX)

        return self.predictions

    def calculate_metrics(self, plot: bool):

        predictions = self.predictions
        real = np.asarray(self.testY_df)

        mae = mean_absolute_error(real, predictions)
        mse = mean_squared_error(real, predictions)

        print('MSE: ', mse, '| MAE: ', mae)

        if plot:
            plt.plot(predictions[0:400], label='Predicted')
            plt.plot(real[0:400], label='Real')

            plt.legend()
            plt.show()

    def train_model(self, model_name: str, X_train, Y_train, epochs, batch_size, validation_split):
        filepath = f"{os.getcwd()}/models/saved_models/{model_name}"

        save_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                          save_weights_only=False, mode='min', period=1)

        early = EarlyStopping(monitor='val_loss', patience=10)

        # Create the model
        model = Sequential()
        model.add(Dense(24, input_shape=(len(X_train.columns),), activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1, activation='linear'))

        # Configure the model and start training
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.005),
                      metrics=['accuracy'])

        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2,
                  callbacks=[save_checkpoint, early])

        return model


def main():
    imputer = DataImputer(path_to_data='..')

    # Process data
    imputer.process_data(test_size=0.33)

    # Train the model
    imputer.model = imputer.train_model('simple_mlp_model.hdf5', imputer.trainX_df, imputer.trainY_df, epochs=50, batch_size=5,
                                  validation_split=0.2)

    y_pred = imputer.test_model_locally(path_to_model='saved_models')
    imputer.calculate_metrics(plot=True)


if __name__ == '__main__':
    main()
