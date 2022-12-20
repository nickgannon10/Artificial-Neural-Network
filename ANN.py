import warnings
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter('ignore', SyntaxWarning)
import numpy as np
import csv

#split data into features and classification
class data_cleaner:
    def __init__(self, x_s, y):
        self.x_s = np.asarray(x_s) / 10 
        self.y = np.asarray(y)

class Artificial_Neural_Network:
    def __init__(self, texfile):
        #ex: [5.1,3.5,1.4,0.2] and [1, 0, 0]
        x_s = []       
        y = []            
        with open(texfile, newline='') as csvfile:
            iris = list(csv.reader(csvfile))
        for i in iris:
            if len(i) != 0:
                if i[-1] == "Iris-setosa":   
                    y.append([1, 0, 0])
                elif i[-1] == "Iris-versicolor":
                    y.append([0, 1, 0])
                elif i[-1] == "Iris-virginica":
                    y.append([0, 0, 1])
                x_s.append(list(map(float, i[:-1])))
        #train, validation, test split -- .6:.2:.2, assuming this is a small data set
        self.training_xy = data_cleaner(x_s[0:30] + x_s[50:80] + x_s[100:130], y[0:30] + y[50:80] + y[100:130])
        self.validation_xy = data_cleaner(x_s[30:40] + x_s[80:90] + x_s[130:140], y[30:40] + y[80:90] + y[130:140])
        self.testing_xy = data_cleaner(x_s[40:50] + x_s[90:100] + x_s[140:150], y[40:50] + y[90:100] + y[140:150])
        #layer to layer node, edge structure, start with 4 features and with 3 outputs  
        np.random.seed(25)    
        self.initial_input_weight = np.full((4,4), 1/16)
        self.initial_hidden_weight = np.full((3,4), 1/12)
        #deal with bias 
        self.bias_weights = np.random.rand(7) 
        self.bias = 1
        self.learning_rate = 0.1   
        self.MSE_goal = 0.1
    def output_activation(self, p):
        #for forward prop, logistic sigmoid
        return 1/(1 + np.exp(-p))
    def derivative_activation_function(self, p):
        #for backprop, chain rule
        return self.output_activation(p) * (1 - self.output_activation(p))

    def Train_Val_Test(self):
        MSE_training = 1
        MSE_val = 1
        while MSE_val >= self.MSE_goal:
            #TRAINING, initialize metrics
            t_squared_error = 0
            correct_count_train = 0
            v_squared_error = 0
            correct_count_val = 0
            for i, y_actual in zip(self.training_xy.x_s, self.training_xy.y):
                #FORWARD PROP, hidden layer potentials and activation function output
                p_hidden = np.dot(self.initial_input_weight, i) + self.bias * self.bias_weights[:4]
                o_hidden = self.output_activation(p_hidden)
                #3 ultimate outputs and 3 potentials
                p_ultimate = np.dot(self.initial_hidden_weight, o_hidden) + self.bias * self.bias_weights[4:]
                y_pred = self.output_activation(p_ultimate)                
                #BACKPROP, determine ultimate error 
                ou_error = self.derivative_activation_function(y_pred) * (y_actual - y_pred)
                #new weights btw ultimate and hidden
                self.initial_hidden_weight += self.learning_rate * np.dot(ou_error.reshape(3,1),o_hidden.reshape(4,1).T)
                self.bias_weights[4:] += self.learning_rate * self.bias * ou_error
                #calc contribution
                oh_error = self.derivative_activation_function(o_hidden) * np.dot(self.initial_hidden_weight.T, ou_error)
                self.initial_input_weight += self.learning_rate * np.dot(oh_error.reshape(4,1), i.reshape(4,1).T)
                self.bias_weights[:4] += self.learning_rate * self.bias * oh_error 
                #sum squared error and correct guesses
                t_squared_error += np.sum((y_actual - y_pred)**2)  
                if y_actual[np.argmax(y_pred)] == 1:
                    correct_count_train += 1
            #VALIDATION
            for i, y_actual in zip(self.validation_xy.x_s, self.validation_xy.y):
                p_hidden = np.dot(self.initial_input_weight, i) + self.bias * self.bias_weights[:4]
                o_hidden = self.output_activation(p_hidden)
                p_ultimate = np.dot(self.initial_hidden_weight, o_hidden) + self.bias * self.bias_weights[4:]
                y_pred = self.output_activation(p_ultimate)                
                v_squared_error += np.sum((y_actual - y_pred)**2)
                if y_actual[np.argmax(y_pred)] == 1:
                    correct_count_val += 1
            MSE_training = t_squared_error / 90
            MSE_val = v_squared_error / 30
            print("Training MSE = ", np.around(MSE_training, decimals=2))
            print("Training Accuracy: ", np.around(correct_count_train / 90 * 100, decimals=2), "%")
            print("Validation MSE = ", np.around(MSE_val, decimals=2),)
            print("Validation Accuracy: ", np.around(correct_count_val/ 30 * 100, decimals=2), "%")        
        #TESTING
        num_correct_test = 0
        for i, y_actual in zip(self.testing_xy.x_s, self.testing_xy.y): 
            p_hidden = np.dot(self.initial_input_weight, i) + self.bias * self.bias_weights[:4]
            o_hidden = self.output_activation(p_hidden)
            p_ultimate = np.dot(self.initial_hidden_weight, o_hidden) + self.bias * self.bias_weights[4:]
            y_pred = self.output_activation(p_ultimate)
            if y_actual[np.argmax(y_pred)] == 1:
                num_correct_test += 1
        print("------------------------------------------")
        print("FINAL RESULTS")
        print("------------------------------------------")
        print("Training Accuracy: ", np.around(correct_count_train / 90 * 100, decimals=2), "%")
        print("Validation Accuracy: ", np.around(correct_count_val/ 30 * 100, decimals=2), "%")
        print("Testing Accuracy: ", np.around(num_correct_test / 30 * 100, decimals=2), "%")
    
    def Trial(self): 
        provided = input("Please provide 4 floats, no commas: ")
        user_provided = np.asarray(list(map(float, provided.split()))) / 10
        p_hidden = np.dot(self.initial_input_weight, user_provided) + self.bias * self.bias_weights[:4]
        o_hidden = self.output_activation(p_hidden)
        p_ultimate = np.dot(self.initial_hidden_weight, o_hidden) + self.bias * self.bias_weights[4:]
        o_ultimate = self.output_activation(p_ultimate)  
        if np.argmax(o_ultimate) == 0: 
            y = "Iris-setosa"
        if np.argmax(o_ultimate) == 1: 
            y = "Iris-versicolor"
        if np.argmax(o_ultimate) == 2: 
            y = "Iris-virginica"
        print("These data suggest, the species is ", y)
        
def main():
    filename = input("Please provide textfile: ")
    #file = "/Users/nicholasgannon/Desktop/AI/A6/dat.txt"
    Neural_net = Artificial_Neural_Network(filename)
    Neural_net.Train_Val_Test()
    Neural_net.Trial()

print(main())