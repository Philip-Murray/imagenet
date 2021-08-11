from model import *


import math

def o(x):
    return 1 / (1 + np.exp(-x))

class BinaryNeuralNetwork(Model):

    def __init__(self, inVectorSize):
        super().__init__(inVectorSize, 2)

        hidden_amt = 156

        self.W1 = 2*np.random.rand(hidden_amt, inVectorSize) - 1
        self.W2 = 2*np.random.rand(1, hidden_amt)

        self.b1 = np.random.rand(hidden_amt)
        self.b2 = np.random.rand(1)


    def predict(self, Xv): #forward pass
        self.X_out = Xv

        self.h_in  = np.matmul(self.W1, Xv)         + self.b1
        self.h_out = o(self.h_in)

        self.y_in  = np.matmul(self.W2, self.h_out) + self.b2
        self.y_out = o(self.y_in)

        if self.y_out < 0.5:
            return 0
        else:
            return 1
      

    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):

        lrn = 0.005
        if X_set.shape[1] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return

        for epoch in range(epochs):
            for b in range(X_set.shape[0]):
                
                self.predict(X_set[b])

                dEdy_out = (Y_set[b] - self.y_out)     #backward pass
                dEdy_in =  np.multiply(dEdy_out, self.y_out * (1-self.y_out))

                dE_dW2 = np.matmul(dEdy_in, np.transpose(self.h_out))
                dE_db2 = dEdy_in * 1

                dEdh_out = np.matmul(np.transpose(self.W2), dEdy_in)
                dEdh_in =  np.multiply(dEdh_out, self.h_out * (1 - self.h_out))

                dE_dW1 = np.matmul(dEdh_in, np.transpose(self.X_out))
                dE_db1 = dEdh_in * 1

                self.W1 -= lrn * dE_dW1
                self.W2 -= lrn * dE_dW2
                self.b1 -= lrn * dE_db1
                self.b2 -= lrn * dE_db2








class MultiClassNeuralNetwork(Model):

    def __init__(self, inVectorSize, outVectorSize):
        super().__init__(inVectorSize, outVectorSize)

        hidden_amt = 156

        self.W1 = 2*np.random.rand(hidden_amt, inVectorSize) - 1
        self.W2 = 2*np.random.rand(outVectorSize, hidden_amt)

        self.b1 = np.random.rand(hidden_amt)
        self.b2 = np.random.rand(outVectorSize)


    def predict(self, Xv): #forward pass
        self.X_out = Xv

        self.h_in  = np.matmul(self.W1, Xv)         + self.b1
        self.h_out = o(self.h_in)

        self.y_in  = np.matmul(self.W2, self.h_out) + self.b2
        self.y_out = o(self.y_in)

        return np.argmax(self.y_out)
      

    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):

        lrn = 0.005
        if X_set.shape[1] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return

        for epoch in range(epochs):
            for b in range(X_set.shape[0]):
                
                self.predict(X_set[b])
                
                one_hot = np.array([1 if v == Y_set[b] else 0 for v in range(self.classifications)])

                dEdy_out = (one_hot - self.y_out)     #backward pass
                dEdy_in =  np.multiply(dEdy_out, self.y_out * (1-self.y_out))

                dE_dW2 = np.matmul(dEdy_in, np.transpose(self.h_out))
                dE_db2 = dEdy_in * 1

                dEdh_out = np.matmul(np.transpose(self.W2), dEdy_in)
                dEdh_in =  np.multiply(dEdh_out, self.h_out * (1 - self.h_out))

                dE_dW1 = np.matmul(dEdh_in, np.transpose(self.X_out))
                dE_db1 = dEdh_in * 1

                self.W1 -= lrn * dE_dW1
                self.W2 -= lrn * dE_dW2
                self.b1 -= lrn * dE_db1
                self.b2 -= lrn * dE_db2

    def dims_assert(self, X_set, Y_set):
        if X_set.shape[0] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return
        if X_set.shape[0] != Y_set.shape[0]:
            print("ERROR - |X_set| != |Y_set|")
            return