# Artificial-Neural-Network
A Neural Network is comprised of 6 parts: 
1. Data Cleaning
    a. the data must be massaged into a suitable set of arrays 
    b. the data must be split into train, validation, and test sets
2. Node Network Construction
    a. 4 inputs given to a four node hidden layer given to 3 node output layer
    b. from layer to layer each node is connected to the other nodes by edges/weights
3. Forward Propagation 
    a. take the dot product of initialized weights and add in a weighted bias
    b. take the output of that function as the input to the sigmoid function
    c. iterate over layers 
4. Backpropagation
    a. for the output layer, determine the difference between the predicted output and the actual output, 
    then integrate rate of change feedback   
    b. for the hidden layer, determine the contribution of a hidden neuron to the error, 
    then integrate rate of change feedback 
5. Validation, Test, and Trial
    a. For validation, test, and trial we are essentially repurposing the hardware built for training, 
    its practically identical to forward Propagation
6. Metrics 
    a. mean squared-error and accuracy scores are provided for the training and validation, just accuracy 
    is reported for testing 

Note for implementation: 
- user input allows you to input your own text file
- the MSE goal value breaks the while loop, it is currently set to .1 but can be lowered if need be,
it will just take longer to find a local minima 
    
    





