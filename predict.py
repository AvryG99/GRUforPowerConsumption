import pandas as pd
import numpy as np
class predict:
    def predict_ahead(self, n_ahead,  x_train, x_test,  model):
        self.yhat = []
        self.n_ahead = n_ahead
        self.x_train = x_train
        self.x_test = x_test
        for _ in range(n_ahead):
        # Making the prediction
            fc = model.predict(x_train)
            self.yhat.append(fc)

        # Creating a new input matrix for forecasting
            x_train = np.append(x_train, fc)

        # Ommitting the first variable
            x_train = np.delete(x_train, 0)

        # Reshaping for the next iteration
            x_train = np.reshape(x_train, (1, len(x_train), 1))
    def make_predict(self):
        ypredict = self.predict_ahead(self.n_ahead, self.x_test[len(self.x_test)-self.n_ahead:])
        yhat = []

        for i in range(self.n_ahead):
            yhat.append(ypredict[0][i][0])
        yhat = np.array(yhat)
        from predata import Data
        data = Data()
        scaler = data.scaler
        yhat = scaler.inverse_transform(yhat.reshape(-1, 1))
