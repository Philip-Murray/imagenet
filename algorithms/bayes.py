from model import *
import numpy as np

class BinaryNaiveBayes(Model):

    def save_to_session(self):
        pass#g_params.session.face_b = self

    def __init__(self, inVectorSize):
        super().__init__(inVectorSize, 2)

        self.prob_xi1_given_y0 = np.ones(self.features)
        self.prob_xi1_given_y1 = np.ones(self.features)
        self.prob_y1 = 1

    def predict(self, Xv):

        #vectorized computation, integer logic
        vector_xi_given_y0 = Xv * self.prob_xi1_given_y0 + (1 - Xv) * (1 - self.prob_xi1_given_y0)
        vector_xi_given_y1 = Xv * self.prob_xi1_given_y1 + (1 - Xv) * (1 - self.prob_xi1_given_y1)

        likelyhood_y0 = (1 - self.prob_y1) * np.prod(vector_xi_given_y0)
        likelyhood_y1 =      self.prob_y1  * np.prod(vector_xi_given_y1)

        if likelyhood_y1 / likelyhood_y0 >= 1:
            return 1
        else:
            return 0        

    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):

        if X_set.shape[1] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return

        cases_y1 = 0

        cases_x1_and_y0 = np.zeros(self.features)
        cases_x1_and_y1 = np.zeros(self.features)

        for b in range(X_set.shape[0]):
            y_act = Y_set[b]
            
            if y_act == 1:
                cases_y1 += 1

            cases_x1_and_y0 += X_set[b] * (1 - y_act)
            cases_x1_and_y1 += X_set[b] * y_act


        self.prob_xi1_given_y0  = cases_x1_and_y0 / (Y_set.shape[0] - cases_y1)
        self.prob_xi1_given_y1  = cases_x1_and_y1 / cases_y1

        self.prob_y1 = cases_y1 / Y_set.shape[0]


    def dims_assert(self, X_set, Y_set):
        if X_set.shape[0] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return
        if X_set.shape[0] != Y_set.shape[0]:
            print("ERROR - |X_set| != |Y_set|")
            return
    



class MultiClassNaiveBayes(Model):

    def save_to_session(self):
        pass#g_params.session.mnist_b = self

    def __init__(self, inVectorSize, outVectorSize):
        super().__init__(inVectorSize, outVectorSize)

        self.matrix_prob_xj1_given_yi1 = None#np.array(outVectorSize, inVectorSize)
        self.vector_prob_Yi            = None#np.array(outVectorSize)


    def predict(self, Xv):
        matrix_xj_given_yi1 = self.matrix_prob_xj1_given_yi1 * Xv + (self.matrix_prob_xj1_given_yi1 - 1) * (1 - Xv)
        
        likelyhood_Yi_vector = self.vector_prob_Yi * np.prod(matrix_xj_given_yi1, axis=1)

        return np.argmax(likelyhood_Yi_vector)      

    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):

        if X_set.shape[1] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return

        vector_cases_yi1 = np.zeros(self.classifications)
        matrix_cases_xi1_and_yi1 = np.zeros(self.classifications * self.features).reshape(self.classifications, self.features)

        for b in range(X_set.shape[0]):
            
            y_act = Y_set[b]
            vector_cases_yi1[y_act] += 1

            matrix_cases_xi1_and_yi1[y_act] += X_set[b]




        self.vector_prob_Yi = vector_cases_yi1 / Y_set.shape[0]
        

        self.matrix_prob_xj1_given_yi1 = np.zeros(self.classifications * self.features).reshape(self.classifications, self.features)

        for Yi in range(self.classifications):
            self.matrix_prob_xj1_given_yi1[Yi] = matrix_cases_xi1_and_yi1[Yi] / vector_cases_yi1[Yi]

        


    def dims_assert(self, X_set, Y_set):
        if X_set.shape[0] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return
        if X_set.shape[0] != Y_set.shape[0]:
            print("ERROR - |X_set| != |Y_set|")
            return

        
