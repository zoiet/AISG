##########################################################################
What is TensorFlow?
    - Open-source library for graph-based numerical computation
    - Low and high level APIs

Defining tensors in TensorFlow
import tensorflow as tf 
# 0D Tensor
d0 = tf.ones((1, ))

# 1D Tensor
d1 = tf.ones((2, ))

# 2D Tensor
d2 = tf.ones((2, 2))

# 3D Tensor
d3 = tf.ones((2, 2, 2))

print(d3.numpy())

A constant is the simplest category of tensor 
    - not trainable
    - can have any dimension

from tensorflow import constant
# Define a 2x3 constant
a = constant(3, shape=[2,3])

# Define a 2x2 constant
b = constant([1,2,3,4], shape=[2,2])
##########################################################################
Defining data as constants

# Import constant from TensorFlow
from tensorflow import constant

# Convert the credit_numpy array into a tensorflow constant
credit_constant = constant(credit_numpy)

# Print constant datatype
print('The datatype is:', credit_constant.dtype)

# Print constant shape
print('The shape is:', credit_constant.shape)

##########################################################################
Defining variables

# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])

# Print the variable A1
print(A1)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()

# Print B1
print(B1)
##########################################################################
Basic Operations

What is a TensorFlow operation?

Applying the addition opeator
# Import constant and add from tensorflow
from tensorflow iport constant, add

# Definee 0-dimensional tensors
A0 = constant([1])
B0 = constant([2])

# Define 1-dimensional tensors
A1 = constant([1, 2])
B1 = constant([3, 4])

# Define 2-dimensional tensors
A2 = constant([1, 2], [3, 4])
B2 = constant([5, 6], [7, 8])

# Perform tensor addition with add()
C0 = add(A0,, B0)
C1 = add(A1, B1)
C2 = add(A2, B2)

The add() operation performs element-wise addition with two tensors
Element-wise addition requires both tensors to have the same shape:
    - Scalar addition
    - Vector addition
    - Matrix addition

Element-wise multiplication multiply() requires both tensors to have the same shape:
Matrix multiplicattion performed with matmul() operator 
    - matmul(A, B)
        Number of columns of A must equal the number of rows of B

# Import operators from tensorflow
from tensorflow import ones, matmul, multiply
# Define tensors
A0 = ones((1)
A31 = ones([3, 1])
A34 = ones([3, 4])
A43 = ones([4, 3])

Valid OOperations: 
    - multiply(A0, A0), multiply(A31, A31), multiply(A34, A34)
    - matmul(A43, A34)
Invalid Operations
    - constant(A43, A43)

Summing over tensor dimensions
    The reduce_sum() oerator sums over the dimentions of a tensor
    reduce_sum(A) sums over all dimensions of A
    reduce_sum(A, i) sus over dimension i
##########################################################################
Performing element-wise multiplication

# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)

# Print the tensors C1 and C23
print('C1: {}'.format(C1.numpy()))
print('C23: {}'.format(C23.numpy()))
##########################################################################
Making predictions with matrix multiplication

# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features, params)

# Compute and print the error
error = bill - billpred
print(error.numpy())
##########################################################################
Advanced Operations

gradient() --> Computes the slope of a function at a point
reshape() --> Reshapes a tensor
random() --> Populates tensor with entries drawn from a probability distribution

Finding the optimum where gradient = 0
Use gradient() to find
    - Minimum: Lowest value of a loss function
    - Maximum: Highest value of objective function

# Import tensorflow under the alias tf
import tensorflow as tf

# Define x
x = tf.Variable(-1.0)
# Define y within instance of GradientTape
with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.multiply(x, x)
# Evaluate the gradient of y at x = -1
g = tape.gradient(y, x)
print(g.numpy())

How to reshape a grayyscale image
# Import tensorflow as alias tf
import tensorflow as tf

# Generate grayscale image
gray = tf.randoom.uniform([2, 2], maxval=255, dtype='int32')

# Reshape grayscale image
gray = tf.reshape(gray, [2*2, 1])

# Generate color image
color = tf.randoom.uniform([2, 2, 3], maxval=255, dtype='int32')

# Reshape color image
color = tf.reshape(color, [2*2, 3])
##########################################################################
Reshaping tensors

# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (784*1, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (2352*1, 1))
##########################################################################
Optimizing with gradients

def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x, x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))
##########################################################################
Working with image data

# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3*1, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())
##########################################################################
Load data using pandas

# Import pandas under the alias pd
import pandas as pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing['price'])
##########################################################################
Setting the data type

# Import numpy and tensorflow with their standard aliases
import numpy as np
import tensorflow as tf

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)
##########################################################################
Loss functions

Fundamental tensorflow operation
    Used to train a model
    Measure of model fit 

High value = worse fit 
    Minimize the loss function 

TensorFlow has operations for common loss functions
    Mean squared eerror (MSE)
    Mean absolute error (MAE)
    Huber error

Loss functions are accessible from tf.keras.losses()
    tf.keras.losses.mse()
    tf.keras.losses.mae()
    tf.keras.losses.Huber()

# Import TensorFlow under standard alias
import tensorflow as tf
# Compute the MSE loss
loss = tf.keras.losses.mse(targets, predictions)
# Define a linear regression model
def linear_regression(intercept, slope = slope, features = features):
    return intercept + features*sope

def loss_function(intercept, slope, targets = targets, features = features):
    # Compute the predictions ffor a linear model
    predictions = linear_regression(intercept, slope)

    # Return the loss
    return tf.keras.losses.mse(targets, predictions)

# Compute the loss for test data inputs
loss_function(intercept, slope, test_targets,test_features)

# Compute the oss for default data inputs
loss_function(intercept, slope)
##########################################################################
Loss functions in TensorFlow

# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)

# Print the mean absolute error (mae)
print(loss.numpy())
##########################################################################
Modifying the loss function

# Initialize a variable named scalar
scalar = Variable(1.0, dtype=float32)

# Define the model
def model(scalar, features = features):
  	return scalar * features

# Define a loss function
def loss_function(scalar, features = features, targets = targets):
	# Compute the predicted values
	predictions = model(scalar, features)
    
	# Return the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())
##########################################################################
Linear regression 

# Define a linear regression model
def linear_regression(intercept, slope, features = size):
    return intercept + features*slope

# Define the prediicted values and loss
def loss_function(intercept, slope, targets = price, features = size) :
    predictions = linear_regression(intercept, slope)
    return tf.keras.losses.mse(targets, predictions)

# Define an optimization operation
opt = tf.keras.optimizers.Adam()

# Minimize the loss function and print the loss
for j in range(1000):
    opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])
    print(loss_function(intercept, slope))

# Print the training parameters
print(intercept.numpy(), slope.numpy())
##########################################################################
Set up a linear regression

# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return intercept + features*slope

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)
    
    # Return the mean squared error loss
	return keras.losses.mse(targets, predictions)

# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())
##########################################################################
Train a linear model

# Initialize an adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(100):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])

	# Print every 10th value of the loss
	if j % 10 == 0:
		print(loss_function(intercept, slope).numpy())

# Plot data and regression line
plot_results(intercept, slope)
##########################################################################
Multiple linear regression

# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)
  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)
##########################################################################
Batch training 
    - Multiple updates per epoch
    - Requires division of dataset
    - No limit on dataset size

Full sample
    - One update per epoch
    - Accepts dataset without modification
    - Limiited by memory

##########################################################################
Preparing to batch train

# Define the intercept and slope
intercept = Variable(10.0, dtype=float32)
slope = Variable(0.5, float32)

# Define the model
def linear_regression(intercept, slope, features):
	# Define the predicted values
	return intercept + features*slope

# Define the loss function
def loss_function(intercept, slope, targets, features):
	# Define the predicted values
	predictions = linear_regression(intercept, slope, features)
    
 	# Define the MSE loss
	return keras.losses.mse(targets, predictions)
##########################################################################
Training a linear model in batches

# Initialize adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
	size_batch = np.array(batch['sqft_lot'], np.float32)

	# Extract the price values for the current batch
	price_batch = np.array(batch['price'], np.float32)

	# Complete the loss, fill in the variable list, and minimize
	opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), var_list=[intercept, slope])

# Print trained parameters
print(intercept.numpy(), slope.numpy())
##########################################################################
Dense layers

import tensorflow as tf
# Define inputs (features)
inputs = tf.constant([[1,35]])
# Define weights
weights = tf.Variable([-0.05], [-0.01]])
# Define the bias
bias = tf.Variable([0..5])
# Multiply imputs (features) by the weights
product = tf.matmul(inputs, weights)
# Define dense layer
dense = tf.keras.activations.sigmoid(product+bias)

# Define inputs (features) layer
inputs = tf.constant([[data, tf.float32]])
# Define dense layer 1
dense1 = tf.keras.layers.Dense(10, activation='sigmoid')(inputs)
# Define dense layer 2
dense2 = tf.keras.layers.Dense(5, activation='sigmoid')(dense1)
# Define otput (predictions) layer
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

High-level approach
    High-level API operations
    dense = keras.layers.Dense(10, activation='sigmoid')
Low-level approach
    Linear-algebraic operations
    prod = matmul(inputs, weights)
    dense = keras.activations.sigmoid(prod)
##########################################################################
The linear algebra of dense layers

# From previous step
bias1 = Variable(1.0)
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1 + bias1)

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)))

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
print('\n prediction: {}'.format(prediction.numpy()[0,0]))
print('\n actual: 1')
##########################################################################
The low-level approach with multiple examples

# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features, weights1)

# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid(products1 + bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print('\n shape of borrower_features: ', borrower_features.shape)
print('\n shape of weights1: ', weights1.shape)
print('\n shape of bias1: ', bias1.shape)
print('\n shape of dense1: ', dense1.shape)
##########################################################################
Using the dense layer operation

# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)
##########################################################################
Activation functions

Sigmoid activation function
    Binary classification
    Low-level: tf.keras.activations.sigmoid()
    High-level: sigmoid

ReLu activation function
    Hiddenn layers
    Low-level: tf.keras.activations.relu()
    High-level: relu

Softmax activation function
    Output layer (> classes)
    Low-level: tf.keras.activations.softmax()
    High-level: softmax

Activation functions in neural networks
import tensorflow as tf
# Define input layer
inputs = tf.constant(borrower_features, tf.float32)

# Define dense layer 1
dense1 = tf.keras.layers.Dense(16, activation='relu')(inputs)

# Define dense layer 2
dense2 = tf.keras.layers.Dense(88, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
outputs = keras.layers.Dense(1, activation='softmax')(dense2)
##########################################################################
Binary classification problems

# Construct input layer from features
inputs = constant(bill_amounts, dtype = float32)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print(error)
##########################################################################
Multiclass classification problems

# Construct input layer from borrower features
inputs = constant(borrower_features, dtype = float32)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)

# Print first five predictions
print(outputs.numpy()[:5])
##########################################################################
optimizers
The gradient descent optimizer
    Sochastic gradient descent (SGD) optimizer
        tf.keras.optimizers.SGD()
        learning_rate
    Simple and easy to interpret

The RMS prop optimizer
    Root mean squared (RMS) propagation optimizer
        Applies different learrning rates to each feature1
        tf.keras.optimizers.RMSprop()
        learning_rate
        momentum
        decay
    Allows for momentum to both buid and decay

The adam optimizer
    Adaptive moment (adam) optimizer
        tf.keras.optimizers.Adam()
        learning_rate
        betal1
    Performs well with default parameter values

# Define the model function
def model(bias, weights, features = borrower_features):
    product = tf.matmul(features, weights)
    return tf.keras.activations.sigmoid(product+bias)

# Compute the predicted values and loss
def loss_function(bias, weights, target = default, features = borrower_features):
    predctions = model(bias, weights)
    return tf.keras.losses.binary_crossentropy(targets, predictions)

# Minimise the loss function wwiith RMS propagation
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
opt.minimize(lambda: loss_function(bias, weights), var_list=[bias, weights])
##########################################################################
The dangers of local minima

# Initialize x_1 and x_2
x_1 = Variable(6.0,float32)
x_2 = Variable(0.3,float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)

for j in range(100):
	# Perform minimization using the loss function and x_1
	opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
	# Perform minimization using the loss function and x_2
	opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())
##########################################################################
Avoiding local minima

# Initialize x_1 and x_2
x_1 = Variable(0.05,float32)
x_2 = Variable(0.05,float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)

for j in range(100):
	opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())
##########################################################################
Trainig a network in TensorFlow

Random initializers
    Often need to initialize thousands of variables
        tf.ones() may perform poorly
        Tedious and difficult to initialize variables individually
    Alternatively, dram initial values from distribution
        Normal
        Uniform

# Define 500x5000 randoom normal variable
weights = tf.Variable(tf.random.normal([500,500]))

# Define 500, 500 truncated random normall variable
weights = tf.Variable(tf.random.truncated_normal([500,500]))

# Define a dense layer with the default initializer
dense = keras.layers.Dense(32, activation='relu')

# Define a dense layer with the zeros initializer
dense = keras.layers.Dense(32, activation='relu', kernel_initializer='zeros')

Implementing dropout in a network

# Define input data
inputs = np.array(borrower_features, np.float32)

# Define first dense layer
dense1 = keras.layers.Dense(32, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(16, activation='relu')(dense1)

# Apply dropout operation
dropout1 = keras.layers.Dropout(0.25)(dense2)

# Define output layer
outputs = keras.layers.Dense(1,activation='sigmoid')(dropout1)
##########################################################################
Initialization in TensorFlow

# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(ones([7]))

# Define the layer 2 weights
w2 = Variable(random.normal([7, 1]))

# Define the layer 2 bias
b2 = Variable(0.0)
##########################################################################
Defining the model and loss function

# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)
##########################################################################
Training neural networks with TensorFlow

# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])

# Make predictions with model
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)
##########################################################################
Building a sequential model

# Define a sequential model
model = keras.Sequential()

# Define first hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(28*28,)))

# Define second hidden layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define second hidden layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Summarize the model
print(model.summary())

Using the functional API
# Define model 1 input layer shape
model1_inputs = keras.Input(shape=(28*28,))

# Define model 2 input layer shape
model2_inputs = keras.Input(shape=(10,))

# Define layer 1 for model 1
model1_layer1 = keras.layers.Dense(12, activation='relu')(model1_inputs)

# Define layer 2 for model 1
model1_layer2 = keras.layers.Dense(4, activation='softmax')(model1_layer1)

# Define layer 1 for model 2
model2_layer1 = keras.layers.Dense(8, activation='relu')(model2_inputs)

# Define layer 2 for model 2
model2_layer2 = keras.layers.Dense(4, activation='softmax')(model2_layer1)

# Merge model 1 and model 2
merged = keras.layers.add([model1_layer2, model2_layer2])

# Define a functional model 
model = keras.Model(inputs=[model1_inputs, model2_inputs], outputs=merged)

# Compile the model
model.compile('adam', loss='categorical_crossentropy')
##########################################################################
The sequential model in Keras 

# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print(model.summary())
##########################################################################
Compiling a sequential model

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')
# Print a model summary
print(model.summary())
##########################################################################
Defining a multiple input model

# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())
##########################################################################
Training and validation with Keras
    1. Load and clean data
    2. Define model 
    3. Train and validate model
    4. Evaluate model

The fit() operation
    Required arguments
        features
        labels
    Many optional arguments
        batch_size
        epochs
        validation_split

# Train model with validation split
model.fit(features, labels, epochs=10, validation_split=0.20)

# Recompile the model with the accuracy metric
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

The evaluation() operation
# Evaluation the test set
model.evaluate(test)

##########################################################################
Training with Keras
# Define a sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)
##########################################################################
Metrics and validation with Keras

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid',  input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)
##########################################################################
Overfitting detection

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))


# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=50, validation_split=0.5)
##########################################################################
Evaluating models

# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)

# Evaluate the small model using the test data
small_test = small_model.evaluate(test_features, test_labels)

# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)

# Evaluate the large model using the test data
large_test = large_model.evaluate(test_features, test_labels)

# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))
##########################################################################
What is Estimators API
    High level submodule
    Less flexible
    Enforces best practices
    Faster Deployment
    Many premade models

Model specification and training
    1. Define feature columns
    2. Load and transform data
    3. Define an Estimators
    4. Apply train operation

# Define a numeric dfeature column
size = tf.feature_column.numeric_column("size")

# Define a categorical feature column
rooms = tf.feature_column.categorical_column_with_vocabulary_list("rooms", ["1", "2", "3", "4", "5"])

# Create dfeature column list
features_list = [size, rooms]

# Define a matrix feature column
features_list = [tf.feature_column.numeric_column('image, shape=(784,')]

# Define input data function
def input_fn():
    #Definie feature dictionary
    features = {"size": [1340,1690, 2720], "rooms": [1,3,4]}
    # Define labels
    labels = [221900, 538000, 180000]
    return features, labels

# Define a deep neural network regression
model0 = tf.estimator.DNNRegressor(feature_column=feature_list, hidden_units=[10,6,6,3])

# Train the regression model
model0.train(input_fn, steps=20)

# Define a deep neural classifier
model1 = tf.estimator.DNNClassifier(feature_column=feature_list, hidden_units=[32,16,8], n_classes=4)

# Train the classifier model
model1.train(input_fn, steps=20)
##########################################################################
Preparing to train with Estimators

# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
	# Define the labels
	labels = np.array(housing['price'])
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels
##########################################################################
Defining Estimators

# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)
##########################################################################

