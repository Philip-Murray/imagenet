from model import *
import numpy as np

class BinaryNaiveBayes(Model):

    def save_to_session(self):
        pass#g_params.session.face_b = self

    def __init__(self, inVectorSize):
        super().__init__(inVectorSize, 2)

        self.prob_xi1_given_y0 = None #np.ones(self.features)
        self.prob_xi1_given_y1 = None #np.ones(self.features)
        self.prob_y1 = None #1

    def predict(self, Xv):

        #vectorized computation, integer logic
        vector_xi_given_y0 = Xv * self.prob_xi1_given_y0 + (1 - Xv) * (1 - self.prob_xi1_given_y0)
        vector_xi_given_y1 = Xv * self.prob_xi1_given_y1 + (1 - Xv) * (1 - self.prob_xi1_given_y1)

        #print(np.prod(vector_xi_given_y0))
        #print(np.prod(vector_xi_given_y1))
        
        #print("MULT "+str(1 - self.prob_y1) + " and "+str(np.prod(vector_xi_given_y0)))
        #print("MULT "+str(self.prob_y1) + " and "+str(np.prod(vector_xi_given_y1)))

        likelyhood_y0 = (1 - self.prob_y1) * np.prod(vector_xi_given_y0)
        likelyhood_y1 =      self.prob_y1  * np.prod(vector_xi_given_y1)

        #print(likelyhood_y0)
        #print(likelyhood_y1)

        if (likelyhood_y1 / likelyhood_y0) >= 1:
            return 1
        else:
            return 0    
        quit()
        if likelyhood_y1 / likelyhood_y0 >= 1:
            return 1
        else:
            return 0   
        quit()
        if (likelyhood_y1 / likelyhood_y0) >= 1:
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

        #print(self.prob_y1)
        #print(self.prob_x1_given_y1)

        #quit()

    def fit2(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):
        #print("FIT)")
        print("FITTIGN)")
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

        #print(self.prob_xi1_y0)
        #print(self.prob_xi1_y1)
        #quit()
    
    def predict2(self, Xv):
        alpha_prob_y0 =     self.prob_y0
        alpha_prob_y1 =     self.prob_y1

        #if(self.prob_y0 == self.prob_y1):
       
        g1 = 1
        g2 = 1

        for f in range(self.features):
            if Xv[f] == 0:
                alpha_prob_y0 *= 1 - self.prob_xi1_y0[f] 
                alpha_prob_y1 *= 1 - self.prob_xi1_y1[f]
                g1 *= 1 - self.prob_xi1_y0[f] 
                g2 *= 1 - self.prob_xi1_y1[f]
            else:
                alpha_prob_y0 *= self.prob_xi1_y0[f]
                alpha_prob_y1 *= self.prob_xi1_y1[f] 
                g1 *= self.prob_xi1_y0[f] 
                g2 *= self.prob_xi1_y1[f]

        #print("MULT "+str( self.prob_y0) + " and "+str(g1))
        #print("MULT "+str(self.prob_y1) + " and "+str(g2))
        #print(g1)
        #print(g2)
        #print("RE")
        #print(alpha_prob_y0)
        #print(alpha_prob_y1)
        #quit()
        #alpha_prob_y0 = self.prob_y0*g1
        #alpha_prob_y1 = self.prob_y1*g2

        print(alpha_prob_y0)
        print(alpha_prob_y1)
        quit()
        if alpha_prob_y1 / alpha_prob_y0 >= 1:
            return 1
        else:
            return 0     


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
        
        likelyhood_Yi_vector =  np.prod(matrix_xj_given_yi1, axis=1) #self.vector_prob_Yi *

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

        
