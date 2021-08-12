from model import *
import numpy as np

class BinaryNaiveBayes(Model):

    def save_to_session(self):
        pass#g_params.session.face_b = self

    def __init__(self, inVectorSize):
        super().__init__(inVectorSize, 2)

        self.prob_xi0_y0 = np.ones(self.features)
        self.prob_xi1_y0 = np.zeros(self.features)
        self.prob_xi0_y1 = np.ones(self.features)
        self.prob_xi1_y1 = np.zeros(self.features)

        self.prob_y0 = 0
        self.prob_y1 = 1

    def predict(self, Xv):
        alpha_prob_y0 = self.prob_y0
        alpha_prob_y1 = self.prob_y1

        for f in range(self.features):
            if Xv[f] == 0:
                alpha_prob_y0 *= self.prob_xi0_y0[f] 
                alpha_prob_y1 *= self.prob_xi0_y1[f]
            else:
                alpha_prob_y0 *= self.prob_xi1_y0[f]
                alpha_prob_y1 *= self.prob_xi1_y1[f] 

        if alpha_prob_y1 / alpha_prob_y0 >= 1:
            return 1
        else:
            return 0        

    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):

        if X_set.shape[1] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return

        cases_y0 = 0
        cases_y1 = 0

        cases_x0_y0 = np.zeros(self.features)
        cases_x1_y0 = np.zeros(self.features)
        cases_x0_y1 = np.zeros(self.features)
        cases_x1_y1 = np.zeros(self.features)

        for b in range(X_set.shape[0]):
            y_act = Y_set[b]
            
            if y_act == 0:
                cases_y0 += 1
            else:
                cases_y1 += 1
                
            for f in range(self.features):
                x_val = X_set[b, f]

                if x_val == 0 and y_act == 0:
                    cases_x0_y0[f] += 1
                if x_val == 1 and y_act == 0:
                    cases_x1_y0[f] += 1
                if x_val == 0 and y_act == 1:
                    cases_x0_y1[f] += 1
                if x_val == 1 and y_act == 1:
                    cases_x1_y1[f] +=1

        self.prob_xi0_y0 = cases_x0_y0 / cases_y0
        self.prob_xi1_y0  = cases_x1_y0 / cases_y0
        self.prob_xi0_y1 = cases_x0_y1 / cases_y1
        self.prob_xi1_y1 = cases_x1_y1 / cases_y1

        self.prob_y0 = cases_y0 / (cases_y0 + cases_y1)
        self.prob_y1 = cases_y1 / (cases_y0 + cases_y1)


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

        self.prob_matrix = None#np.array(outVectorSize, inVectorSize)
        self.prY_vector  = None#np.array(outVectorSize)


    def predict(self, Xv):

        alpha_prY_vector = np.copy(self.prY_vector)

        for f in range(self.features):
            if Xv[f] == 0:
                alpha_prY_vector = np.multiply(alpha_prY_vector, 1 - self.prob_matrix[:, f])
            else:
                alpha_prY_vector = np.multiply(alpha_prY_vector, self.prob_matrix[:, f])


        return np.argmax(alpha_prY_vector)      

    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):

        if X_set.shape[1] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return

        cases_yi = np.zeros(self.classifications)
        cases_matrix_xi_and_yi = np.zeros(self.classifications * self.features).reshape(self.classifications, self.features)

        for b in range(X_set.shape[0]):
            y_act = Y_set[b]            
            cases_yi[y_act] += 1

            for f in range(self.features):
                if X_set[b, f] == 1:
                    cases_matrix_xi_and_yi[y_act, f] += 1



        self.prob_matrix = cases_matrix_xi_and_yi
        self.prY_vector = cases_yi / self.classifications
        
        for row in range(self.classifications):
            self.prob_matrix[row] = self.prob_matrix[row] / cases_yi[row]

        


    def dims_assert(self, X_set, Y_set):
        if X_set.shape[0] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return
        if X_set.shape[0] != Y_set.shape[0]:
            print("ERROR - |X_set| != |Y_set|")
            return

        
