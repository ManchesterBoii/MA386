"""
This is our Artificial Neural Network Module
"""
#Import the necessary modules
import numpy as np
import math
from itertools import combinations
from KFoldCrossVal import foldsToTrainAndTest


#This is the activation function and its derivative intialized as lambda functions. These will be used in the forward and backwards propagation methods.
sigmoid = lambda x : 1/(1+np.exp(-x))
sig_der = lambda x : sigmoid(x)*(1-sigmoid(x))

def weight_bias_init_Nlayers(Size_of_nlayers_lst):
    """
    A function that initializes the weights for each layer of the neural network and also initiates a bias. These will be adjusted later by the neural network.
    Inputs: -Size_of_nlayers_lst: a list of size n that contains the initial value for each neuron in every layer of the network. (n represents the number of neurons in the layer)
    Outputs: - Weights: A list of length n-1 that contains the individual values for each weight
             - Bias: A list of length n-1 that contains the individual values for each bias
    """
    Weights = []
    Bias = []
    for i in range(1,len(Size_of_nlayers_lst)):
        Weights.append(np.random.normal(scale = 0.1, size = (Size_of_nlayers_lst[i],Size_of_nlayers_lst[i-1])))
        Bias.append(np.random.normal(scale = 0.1, size = (Size_of_nlayers_lst[i],1)))
    return Weights, Bias

def forward_propagation(X,W,b):
    """
    A helper function that performs forward propagation on a dataframe column.
    Inputs: -X: a row of data from a pandas dataframe
            -W: a list of weight values given to each attribute
    Outputs: -Activation: A vector of activated values that can be passed on to the next layer of the network or to the final output of the network
             -fwd_comp: The dot product of the weights and the row of data, with the bias added after
    """
    fwd_comp = np.dot(W,X) + b # forward computation of the linear function Y= Wx+b
    Activation = sigmoid(fwd_comp) # sigmoid activation fo the linear fucntion
    return Activation, fwd_comp

def N_layer_FW_Prop(X, W, B, N_layer):
    """
    A function that performs all of the forward propagation steps for the neural network
    Inputs: -X: a row of data from the pandas dataframe
            -W: A vector of length n-1 (where n is the length of X) containing the weights for each feature
            -B: The bias value (assumed to be 0)
            -N_layer: The number of layers in the neural network
    Outputs: -activation: a vector containing the activation values for the specific layer of the network
             -Activation_list: a list of lists containing all the activation values as they were adjusted through forward propagation
             -fwd_computations: a list of the dot products between the weights and rows of data, with each value appearing as each row was passed through
    """
    Activation_list = []
    fwd_computations  = []
    curr_activation = X
    for i in range(N_layer-1):
        W_i = np.copy(W[i])
        activation, fwd_comp = forward_propagation(curr_activation, W_i, B[i]) 
        Activation_list.append(activation)
        fwd_computations.append(fwd_comp)
        curr_activation = activation
    return activation, Activation_list, fwd_computations

def N_layer_BWD_Prop(Y, X, W, N_layer, prev_Activation, Activationlst, fwdcomputations):
    """
    A function that performs backward propagation on a data set with a neural network of n layers, where n is the length of the dataset.
    Gradient Decent with respect to all of the input parameters
    Inputs: -Y: the column of actual values for our response variable (in this case the column of 0s and 1s titled "Graduate")
            -X: The entire dataframe containing the explanatory variables 
            -N_layer: The number of layers with in the neural network, initiated as the number of rows in the dataset
            -prev_Activation: Our yhat vector
            -Activationlst: a list of all the activation values as they were adjusted through forward propagation
            -fwdcomputations: a list of the dot products between the weights and rows of data, with each value appearing as each row was passed through
    Outputs: gradient: a list of values of the gradients for the activation function as calculated through the backwards propagation process 
    """
    Y = Y.reshape(prev_Activation.shape)
    m = prev_Activation.shape[1]
    gradient = []
    deltaA = -(Y-prev_Activation) # loss at the output layer layer
    dA_curr = deltaA
    for i in reversed(range(1,N_layer-1)):
        DZ = dA_curr * sig_der(fwdcomputations[i]) # DZ is the product of derivative of the activation of the forward linear computation and the loss of the previous layer
        A_prev = Activationlst[i-1] # Activation of the previous layer
        dw = (1/m)*np.dot(dZ,A_prev.T) # Derivative of the activation at the previous layer with respect to dz produces the gradient of the weights at the current layer
        db = (1/m)*np.sum(dZ, axis=1, keepdims=True) #Gradient of the Bias
        gradient.append((dw,db))
        dA_prev = np.dot(W[i].T, DZ)  # loss at previous layer
        dA_curr = dA_prev
        
    # back propagation at the input layer
    dZ = dA_curr * sig_der(fwdcomputations[0])
    A_prev = X  #to get the gradient decent of the weights at the input layer we must us the inputs at the input layer as the final "activation"
    dw_curr = (1/m)*np.dot(dZ,A_prev.T)
    db_curr = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    gradient.append((dw_curr,db_curr))
    return gradient

def N_layer_weights_update(learning_rate, gradients, W,b,N_layer):
    """
    A function that updates the weights for each layer of the neural network
    Inputs: -learning_rate: the learning rate of the neural network, set by the user
            -gradients: the list of gradients as calculated through the backwards propagation process 
            -W: A vector containing the values for each weight 
            -b: The bias value
            -N_layer: the number of layers in the neural network
    Outputs: -W: a list containing the adjusted values for each weight
             -b: the adjusted bias value
    """
    for i in range(N_layer-1):
        W[i] = W[i] - learning_rate*gradients[i][0]  #updating the weight based on the gradient loss and the learning rate
        b[i] = b[i] - learning_rate*gradients[i][1]  #updating the nias based on the gradient loss and the learning rate
    return W, b

def model_predict(X,W,b,thresh,n):
    """
    A function that creates the prediction values for the final model
    Inputs: -X: The dataframe
            -W: The vector of calculated weights
            -b: The bias value
            -thresh: The threshold value
            -n: number of layers in the neural network
    Outputs: -y_out: The final activation value for the nth level of the neural network
             -y_pred: The predicted value as estimated by the neural network
    """
    y_pred = []
    y_out = N_layer_FW_Prop(X, W, b, n)[0][0]
    for y in y_out:
        if y < thresh:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_out, y_pred

def train_NN(Dataframe, epoch, learning_rate, layer_neurons):
    """
    The culminating function of this module. This function accepts a dataframe, a response column, an epoch value, and a learning rate. It crunches the dataframe and calculates the final weight each variable should be given and the final bias. Used for k fold cross validation.
    Inputs: -Dataframe: The dataframe
            -epoch: The number of times to send the data through the neural network to adjust the weights
            -learning_rate: the value to adjust the weights by per iteration in the network
            -layer_neurons: a list of how many hidden layers you want and how many neurons for each layer
    Outputs: -Acc_C: List of accuracies for each fold
             -Ws_C: List of final wieghts for each fold
             -Cost_C: List of SSE for each fold
             -Untouched_data: Data that the Neural network has not seen
    """
    selects = np.random.rand(len(Dataframe)) < 0.6
    Combos = foldsToTrainAndTest(Dataframe[selects])
    Ws_C = []
    Cost_C= []
    acc_C = []
    for idx,p in enumerate(Combos):
        traindata = p[0]
        testdata = p[1]
        tempcost = []
        X = traindata[:,0:-1].T
        layerlist = [X.shape[0]] + layer_neurons + [1]
        Y = traindata[:,-1]
        Ws, Bs = weight_bias_init_Nlayers(layerlist)
        for i in range(epoch):
            Yhat, Activationlist, fwdcomputations = N_layer_FW_Prop(X, Ws, Bs, len(layerlist)) # forward step
            gradient = N_layer_BWD_Prop(Y,X, Ws, len(layerlist), Yhat , Activationlist ,fwdcomputations) #backwards step
            cost = ((Yhat-Y)**2).sum() #calculate cost SSE
            tempcost.append(cost)
            Ws, Bs = N_layer_weights_update(learning_rate, gradient[::-1], Ws, Bs, len(layerlist)) #update layers
            print(f'{(i/(epoch*len(Combos))+(idx)*0.2)*100:.2f}% complete')
        Cost_C+= [tempcost]
        Ws_C.append((Ws, Bs))
        Xtest = testdata[:,0:-1].T
        Ytest = testdata[:,-1]
        y_prob, y_pred = model_predict(Xtest, Ws, Bs, 0.5, len(layerlist)) #predict testdata
        acc_C.append(np.mean(y_pred==Ytest))
    print('Done')
    untouched_data = Dataframe[~selects]
    return acc_C, Ws_C, Cost_C, untouched_data




def train_NN_final_model(X,Y, epoch, learning_rate, X_test, Y_test, layer_neuron ):
    '''
    The culminating function of this module. Used to compute final weights.
    Inputs: -X: Observations that NN is training on 
            -Y: Predictions for the observations that the NN is Training on 
            -epoch: The number of times to send the data through the neural network to adjust the weights
            -learning_rate: the value to adjust the weights by per iteration in the network
            -X_test: The Obsevations that the NN will be making predictions with
            -Y_test: The Predicitions that The NN will be making Predictions aigainst
            -layer_neurons: a list of how many hidden layers you want and how many neurons for each layer
    Outputs: -Ws: A list of values containing the final calculated values of weights
             -Bs: The final value of the bias
             -number of layers in the Neural network
             -testerr: list of Prediction errors
             -costs: list of SSE of the Training data
    '''
    testerr = []
    costs = []
    layerlist = [X.shape[0]] + layer_neuron + [1]
    Ws, Bs = weight_bias_init_Nlayers(layerlist)
    for i in range(epoch):
        print(f"{(i/epoch)*100:.2f}% Complete")
        Yhat, Activationlist , fwdcomputations = N_layer_FW_Prop(X, Ws,Bs, len(layerlist)) #froward Step
        gradient = N_layer_BWD_Prop(Y,X, Ws, len(layerlist), Yhat ,Activationlist ,fwdcomputations)#backwar proagration 
        cost = ((Yhat-Y)**2).sum() #calculate cost SSE
        costs.append(cost)
        Ws, Bs = N_layer_weights_update(learning_rate, gradient[::-1], Ws, Bs, len(layerlist))

        testpred =  model_predict(X_test,Ws , Bs,.5, len(layerlist))[0] #predict testdata
        err = ((testpred-Y_test)**2).sum() #predict err
        testerr.append(err)
        if testerr:
            if err > testerr[-1]:
                break
            if abs(testerr[-1] - err) > 0.005:
                break
    return Ws, Bs, len(layerlist), testerr, costs

