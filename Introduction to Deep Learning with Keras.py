################################################################################
What is Keras
    - Deep Learning Framework
    - Fast industry-ready models
    - Less code
    - Keras is complementary to TensorFlow
    - Can use TensorFlow for low level features

Machine Learning: Inputt > Fature extraction > Classification > Output
Deep Learning: Inut > Feature extraction + Classification > Output

When to use neural networks
    - Dealing with unstructed data
    - Do not need easilyy interpertable results

# Defining a neural networks
from keras.models import Sequential
from keras.layers import Dense

# Create a new sequential model 
model = Sequential()
# Add and input and dense layer
model.add(Dense(2, input_shape=(3,), activation="relu"))
# Add a final 1 neuron layer
model.add(Dense(1))

model.summary()


################################################################################
Hello nets!

# Import the Sequential model and Dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()


# Add an input layer and a hidden layer with 10 neurons
model.add(Dense(10, input_shape=(2,), activation="relu"))

# Add a 1-neuron output layer
model.add(Dense(1))

# Summarise your model
model.summary()
################################################################################
Build as shown

from keras.models import Sequential
from keras.layers import Dense

# Instantiate a Sequential model
model = Sequential()

# Build the input and hidden layer
model.add(Dense(3, input_shape=(2,), activation="relu"))

# Add the ouput layer
model.add(Dense(1))
################################################################################
Surviving a meteor strike
Specifying a model

# Instantiate a Sequential model
model = Sequential()

# Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, input_shape=(1,), activation='relu'))

# Add two Dense layers with 50 neurons and relu activation
model.add(Dense(50, input_shape=(1,), activation='relu'))
model.add(Dense(50, input_shape=(1,), activation='relu'))

# End your model with a Dense layer and no activation
model.add(Dense(1))
################################################################################
Training

# Compile your model
model.compile(optimizer = 'adam', loss = 'mse')

print("Training started..., this can take a while:")

# Fit your model on your data for 30 epochs
model.fit(time_steps,y_positions, epochs = 30)

# Evaluate your model 
print("Final lost value:",model.evaluate(time_steps,y_positions))
################################################################################
Predicting the orbit!

# Predict the eighty minute orbit
eighty_min_orbit = model.predict(np.arange(-40, 41))

# Plot the eighty minute orbit 
plot_orbit(eighty_min_orbit)
################################################################################
Binary Classification

Pairplots
import seaborn as sns
# Plot a pairplot
sns.pairplot(circles, hue="target")

from keras.models import Sequential
from keras.layers import Dense

# Instantiate a Sequential model
model = Sequential()

# Build the input and hidden layer
model.add(Dense(4, input_shape=(2,), activation="tanh"))
# Add output layer, use sigmoid
model.add(Dense(1,activation="sigmoid"))

# Compile model
model.compile(optimizer='sgd', loss='binary_crossentropy')
# Train model
model.train(coordinates, labels, epochs=20)
# Predict with trained model
preds = model.predict(coordinates)
################################################################################
Exploring dollar bills

# Import seaborn
import seaborn as sns

# Use pairplot and set the hue to be our class
sns.pairplot(banknotes, hue='class') 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations of each class
print('Observations per class: \n', banknotes['class'].value_counts())
################################################################################
A binary classification model

# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a sequential model
model = Sequential()

# Add a dense layer 
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model.summary()
################################################################################
Is this dollar bill fake ?

# Train your model for 20 epochs
model.fit(X_train, y_train, epochs=20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:',accuracy)
################################################################################
Multi-class classification

Categorical cross-entroy
model.compile(optimizer='adam', loss='categorical_crossentropy')

import pandas as pd
from keras.utils import to_categorical

# Load dataset
df = pd.read_csv('data.csv')
# Turn response variable into labeled codes
df.response = pd.Categorical(df.response)
df.response = df.response.cat.codes
# Turn response variable into one-hot response vector
y = to_categorical(df.response)

################################################################################
A multi-class model

# Instantiate a sequential model
model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
################################################################################
Prepare your dataset

# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 

# Import to_categorical from keras utils module
from keras.utils import to_categorical

# Use to_categorical on your labels
coordinates = darts.drop(['competitor'], axis=1)
competitors = to_categorical(darts.competitor)

# Now print the to_categorical() result
print('One-hot encoded competitors: \n',competitors)
################################################################################
Training on dart throwers

# Train your model on the training data for 200 epochs
model.fit(coord_train,competitors_train,epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)[1]

# Print accuracy
print('Accuracy:', accuracy)
################################################################################
Softmax predictions

# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

# Extract the indexes of the highest probable predictions
preds = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print("{:10} | {}".format('Rounded Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{:25} | {}".format(pred,competitors_small_test[i]))
################################################################################
Multi-label classification

Use sigmoid ouput
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='sigmoid'))
  
# Compile your model using binary_crossentropy loss
model.compile(loss='binary_crossentropy', optimizer='adam')

# Train your model, recall validation_split
model.fit(X_train, y_train, epochs=100, validation_splits=0.2)
            
################################################################################
An irrigation machine

# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64, input_shape=(20,), activation='relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(3, activation='sigmoid'))

# Compile your model with adam and binary crossentropy loss
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()
################################################################################
Training with multiple labelss

# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs = 100, validation_split = 0.2)

# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)[1]

# Print accuracy
print('Accuracy:', accuracy)
################################################################################
Keras callbacks

# Import early stopping from keras callbacks
from keras.callbacks import EarlyStopping

#Instantiate an early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train your model with the callback
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Import model checkpoint from keras callbacks
from keras.callbacks import ModelChheckpoint

# Instantiate a model checkpoint callback
model_save = ModelChecpoint('best_model.hdf5', save_best_only=True)

# Train your model with the callback
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks = [model_save])

################################################################################
The history callback

# Train your model and save its history
history = model.fit(X_train, y_train, epochs = 50,
               validation_data=(X_test, y_test))

# Plot train vs test loss during training
plot_loss(history.history['loss'], history.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(history.history['acc'], history.history['val_acc'])
################################################################################
Early stopping your model

# Import the early stopping callback
from keras.callbacks import EarlyStopping

# Define a callback to monitor val_acc
monitor_val_acc = EarlyStopping(monitor='val_acc', 
                       patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, 
           epochs=1000, validation_data=(X_test, y_test),
           callbacks=[monitor_val_acc])
################################################################################
A combination of callbacks

# Import the EarlyStopping and ModelCheckpoint callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor = 'val_acc', patience=3)

# Save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only = True)

# Fit your model for a stupid amount of epochs
history = model.fit(X_train, y_train,
                    epochs = 10000000,
                    callbacks = [monitor_val_acc, modelCheckpoint],
                    validation_data = (X_test, y_test))
################################################################################
Learning curve

# Store initial model weights
init_weights = model.get_weights()

# Lists for storing accuracies
train_accs = []
tests_accs = []

for train_size in train_sizes:
    #Spllit a fraction accordinig to train_sizee
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_train, train_size=train_size)

    # Set modell initial weights
    model.set_weights(initial_weights)
    # Fit model on the training set fraction
    model.fit(X_train_frac, y_train_frac, epochs=100, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=1)])
    # Get the accuracy for this training set fraction
    train_acc = model.evaluate(X_train_frac, y_train_frac, verbose=0)[1]
    train_accs.append(train_acc)
    # Get the accuracy on the whole test set
    test_acc = model.evaluate(X_test, y-test, verbose=0)[1]
    test_accs.append(test_acc)
    print("Done with size: ", train_size)
################################################################################
Learning the digits

# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(Dense(16, input_shape = (64,), activation = 'relu'))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10, activation = 'softmax'))

# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Test if your model works and can process input data
print(model.predict(X_train))
################################################################################
Is the model overfitting?

# Train your model for 60 epochs, using X_test and y_test as validation data
history = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=0)

# Extract from the history object loss and val_loss to plot the learning curve
plot_loss(history.history['loss'], history.history['val_loss'])
################################################################################
Do we need more data?

for size in training_sizes:
  	# Get a fraction of training data (we only care about the training data)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new data fraction
    model.set_weights(initial_weights)
    model.fit(X_train_frac, y_train_frac, epochs = 50, callbacks = [early_stop])

    # Evaluate and store the train fraction and the complete test set results
    train_accs.append(model.evaluate(X_train_frac, y_train_frac)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])
    
# Plot train vs test accuracies
plot_results(train_accs, test_accs)
################################################################################
Activation functions

# Set a random seed
np.random.seed(1)
# Return a new model with the given activation
def get_model(act_function):
    model = Sequencial()
    model.add(Dense(4, input_shape=(2,), activation=act_function))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Activation functions too try out
activations = ['relu', 'sigmoid', 'tanh']

# Dictionary to store results
actvation_results = {}
for funct in activations:
    model = get_modell(act_function=funct)
    history = model.fit(X_train y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
    activation_results[funct] = history
################################################################################
Comparing activation functions

# Activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}

for act in activations:
  # Get a new model with the current activation
  model = get_model(act_function=act)
  # Fit the model
  history =  model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
  activation_results[act] = history
################################################################################
Comparing activation functions II

# Create a dataframe from val_loss_per_function
val_loss= pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()
################################################################################
Batch size and batch normalization

Mini batches
    Advantages
        - Networks train faster (more weight updates in same amount of time)
        - Less RAM memory required, can train on huge datasets
        - Noise can help networks reach a lower error, escaping local minima
    Disadvantages
        - More iterations need to be run 
        - Need to be adjusted we need to fiind a good batch size

Batch normalization Advantages
    - Improves gradient flow 
    - Allows higher learning rates
    - Reduces dependence on weighht initializations
    - Acts as an unintended form of regularization
    - Limits internal covariate shift

# Iport BatchNormalization from keeras layers
from keras.layers import BatchNormalization

# INstantiate a Sequential model
model = Sequential()
# Add an input layer
model.add(Dense(3, input_shape=(2,), activation='relu'))
# Add batch normalization foor the outputs of the layer above
model.add(BatchNormalization())
# Add an output layer
model.add(Dense(1, activation='sigmoid'))
################################################################################
Changing batch sizes

model = get_model()

# Fit your model for 5 epochs with a batch of size the training set
model.fit(X_train, y_train, epochs=5, batch_size=X_train.shape[0])
print("\n The accuracy when using the whole training set as a batch was: ",
      model.evaluate(X_test, y_test)[1])
################################################################################
Batch normalizing a familiar model

# Import batch normalization from keras layers
from keras.layers import BatchNormalization

# Build your deep network
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
################################################################################
Batch normalization effects

# Train your standard model, storing its history
history1 = standard_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, verbose=0)

# Train the batch normalized model you recently built, store its history
history2 = batchnorm_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, verbose=0)

# Call compare_acc_histories passing in both model histories
compare_histories_acc(history1, history2)
################################################################################
Hyperparameter tuning

Neural network hyperparameters
    Number of layers
    Number of neurons per layer
    Layer order
    Layer activations
    Batch sizes
    Learning rates
    Optimizers

Turn a Keras model into a Sklearn estimator
# Function that creates out Keras model
def create_model(optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(Dense(16, input_shape=(2,), activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

# Import sklearrn wrapper from keras
from keras.wrappers.scikit_learn import KerasClassifier

# Create a model as a sklearn estimator
model = KerasClassifier(build_fn=create_model, epochs=6, batch_size=16)

# Import cross_val_score
from sklearn.model_selection import cross_val_score

# Check how youe keras model performs with 5 fold crossvalidation
kfold = cross_val_score(model, X, y, cv=5)

# Print the mean accuracy per fold
kfold.mean()

# Print the std per fold
kfold.std()

Tips for neural networks hyperparameter tuningg
    Random search is preferred over grid search
    Do not use many epochs
    Use a smaller sample of your dataset
    Play with batch sizes, activations, optimizers and learning rates

# Define a series of parameters
params = dict(optimizer=['sgd', 'adam']), epochs=3, batch_size=[5, 10, 20], activation=['relu', 'tanh'])

# Create a random search cv object and fit it to the data
random_search = RandomizedSearchCV(model, params_dist-params, cv=3)
random_search_results = random_search.fit(TX, y)

# Print results
print("Best: %f using %s".format(random_search_results.best_score_, random_search_results.best_params_))
################################################################################
Preparing a model for tuning

# Creates a model given an activation and learning rate
def create_model(learning_rate=0.01, activation='relu'):
  
  	# Create an Adam optimizer with the given learning rate
  	opt = Adam(lr=learning_rate)
  	
  	# Create your binary classification model  
  	model = Sequential()
  	model.add(Dense(128, input_shape=(30,), activation=activation))
  	model.add(Dense(256, activation=activation))
  	model.add(Dense(1, activation='sigmoid'))
  	
  	# Compile your model with your optimizer, loss, and metrics
  	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  	return model
################################################################################
Tuning the model parameters

# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256], 
          'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions = params, cv = KFold(3))

# Running random_search.fit(X,y) would start the search,but it takes too long! 
show_results()
################################################################################
Training with cross-validation

# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model, epochs = 50, 
             batch_size = 128, verbose = 0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv=3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())
################################################################################
Tensors, layers, and autoencoders

# Import Keras backend
import keras.backend as K
# Get the input and output tensors of a model layer
inp = model.layers[0].input
out = model.layers[0].output

# function that maps layer inputs to outputs
inp_to_out = K.function([inp], [out])
# We pass and input and get the output we get in that first layer
print(inp_to_out([X_train]))

Autoencoder user case 
    Dimensionality reduction:
        Smaller dimensional space representation of our inputs
    De-noising data
        If trained with clean data, irrelevant noisee will be filtered out during reconstruction
    Anomaly detection
        A poor reconstruction will result when the model is fed with unseen inputs 

Building a simple autoencoder 
# Instantiate a sequential model
autoencoder = Sequential()
# Add a hidden layer of 4 neurons and an input layer of 100
autoencoder.add(Dense(4, input_shape=(100,), activation='relu'))
# Add an output layer of 100 neurons
autoencoder.add(Dense(100, activation='sigmoid'))

# Building a seperate model to encode inputs
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Predicting returns the four hidden layer neuron outputs
encoder.predict(X_test)

################################################################################
It's a flow of tensors

# Import keras backend
import keras.backend as K

# Input tensor from the 1st layer of the model
inp = model.layers[0].input

# Output tensor from the 1st layer of the model
out = model.layers[0].output

# Define a function from inputs to outputs
inp_to_out = K.function([inp], [out])

# Print the results of passing X_test through the 1st layer
print(inp_to_out([X_test]))
################################################################################
Neural separation
Building an autoencoder
for i in range(0, 21):
  	# Train model for 1 epoch
    h = model.fit(X_train, y_train, batch_size=16, epochs=1,verbose=0)
    if i%4==0: 
      # Get the output of the first layer
      layer_output = inp_to_out([X_test])[0]
      
      # Evaluate model accuracy for this epoch
      test_accuracy = model.evaluate(X_test, y_test)[1] 
      
      # Plot 1st vs 2nd neuron output
      plot()
################################################################################
De-noising like an autoencoder
# Build your encoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Encode the images and show the encodings
preds = encoder.predict(X_test_noise)
show_encodings(preds)

# Predict on the noisy images with your autoencoder
decoded_imgs = autoencoder.predict(X_test_noise)

# Plot noisy vs decoded images
compare_plot(X_test_noise, decoded_imgs)
################################################################################
Intro to CNNs

# Import Conv2D layer and Flatten from keras layers
from keras.layers import Dense, Conv2D, Flatten

# Instantiate your model as usual
model = Sequential()

# Add a concolutional layer with 32 filters of size 3x3
moddel.add(Conv2D(filters=32, kernal_size=3, input_shape=(28, 28, 1), activation='relu'))
# Add another convolutional layer
model.add(Conv2D(8, kernal_size=3, activation='relu'))

#Flatten the output of the previous layer
model.add(Flatten())

# End this multiclass model with 3 outputs and softmax
model.add(Dense(3, activation='softmax'))

# Import image from keras preprocessing
from keras.preprocessing import image

# Import preprocessing_input from keras applications resnet50
from keras.applications.resnet50 import preprocessing_input

# Load the image with the right target size for your model 
img = image.load_img(img_path, target_size=(224, 224))

# Turn it into an array
img = image.img_to_array(img)

# expand the dimensions so that it's understood by our network:
# img.shape turns from (224, 224, 3) into (1, 224, 224, 3)
img = np.expand_dims(img, axis=0)

# Pre-process the img in the same way training images were
img = preprocess_input(img)

# Import ResNet50 and decode_predictions from keras.applications.renet50
from keras.applications.resnet50 import RsNet50, decode_predictions

# Instantiate a ResNet50 model with imagenet weights
model = ResNet50(weights='imagenet')

# Predict with ResNet50 on our img
preds = model.predict(img)

# Decode predictions and print it
print('Predicted:', decode_predictions(preds, top=1)[0])
################################################################################
Building a CNN model

# Import the Conv2D and Flatten layers and instantiate model
from keras.layers import Conv2D, Flatten
model = Sequential()

# Add a convolutional layer of 32 filters of size 3x3
model.add(Conv2D(filters=32, input_shape=(28, 28, 1), kernel_size=3, activation='relu'))

# Add a convolutional layer of 16 filters of size 3x3
model.add(Conv2D(filters=16, input_shape=(28, 28, 1), kernel_size=3, activation='relu'))

# Flatten the previous layer output
model.add(Flatten())

# Add as many outputs as classes with softmax activation
model.add(Dense(10, activation='softmax'))
################################################################################
Looking at convolutions

# Obtain a reference to the outputs of the first layer
layer_output = model.layers[0].output

# Build a model using the model's input and the first layer output
first_layer_model = Model(inputs = model.input, outputs = layer_output)

# Use this model to predict on X_test
activations = first_layer_model.predict(X_test)

# Plot the activations of first digit of X_test for the 15th filter
axs[0].matshow(activations[0,:,:,14], cmap = 'viridis')

# Do the same but for the 18th filter now
axs[1].matshow(activations[0,:,:,17], cmap = 'viridis')
plt.show()
################################################################################
Preparing your input image

# Import image and preprocess_input
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# Load the image with the right target size for your model
img = image.load_img(img_path, target_size=(224, 224))

# Turn it into an array
img_array = image.img_to_array(img)

# Expand the dimensions of the image
img_expanded = np.expand_dims(img_array, axis = 0)

# Pre-process the img in the same way original images were
img_ready = preprocess_input(img_expanded)
################################################################################
Using a real world model

# Instantiate a ResNet50 model with 'imagenet' weights
model = ResNet50(weights='imagenet')

# Predict with ResNet50 on your already processed img
preds = model.predict(img_ready)

# Decode the first 3 predictions
print('Predicted:', decode_predictions(preds, top=3)[0])
################################################################################
Intro to LSTMs

text = 'Hi this is a small sentence'
# We choose a sequence length
seq_len = 3

# Split text into a list of words
words = text.split()

# Make lines
lines = []

for i in range(seq_len, len(words) + 1):
    line = ' '.join(words[i-seq_len:i])
    llines. append(line)

# Import Tokenizer from keras preprocessing text
from keras.preprocessing.text import Tokenizer

# Instantiate Tokenizer
tokenizer = Tokenizer()

# Fit it on the previous lines
tokenizer.fit_on_texts(lines)

# Turn the lines into numeric sequences
sequences = tokenizer.texts_to_sequences(lines)

print(tokenizer.index_word)

# Import Dense, LSTM and Embedding layers
from keras.layers import Dense, LSTM, Embedding
model = Sequential()
# Vocabulary size
vocab_size = len(tokenizer.index_word) + 1
# Starting with an embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=2))
# Adding an LSTM layer
model.add(LSTM(8))

# Adding a Dense hidden layer
model.add(Dense(8, activation='relu'))
# Adding an output layer with softmax
model.add(Dense(vocab_size, activation='softmax'))
################################################################################
Text prediction with LSTMs

# Split text into an array of words 
words = text.split()

# Make lines of 4 words each, moving one word at a time
lines = []
for i in range(4, len(words)):
  lines.append(' '.join(words[i-4:i]))

# Instantiate a Tokenizer, then fit it on the lines
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)

# Turn lines into a sequence of numbers
sequences = tokenizer.texts_to_sequences(lines)
print("Lines: \n {} \n Sequences: \n {}".format(lines[:5],sequences[:5]))
################################################################################
Build your LSTM model

# Import the Embedding, LSTM and Dense layer
from keras.layers import Dense, LSTM, Embedding

model = Sequential()

# Add an Embedding layer with the right parameters
model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=3))

# Add a 32 unit LSTM layer
model.add(LSTM(32))

# Add a hidden Dense layer of 32 units and an output layer of vocab_size with softmax
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()
################################################################################
Decode your predictions

def predict_text(test_text):
  if len(test_text.split())!=3:
    print('Text input should be 3 words!')
    return False
  
  # Turn the test_text into a sequence of numbers
  test_seq = tokenizer.texts_to_sequences([test_text])
  test_seq = np.array(test_seq)
  
  # Get the model's next word prediction by passing in test_seq
  pred = model.predict(test_seq).argmax(axis = 1)[0]
  
  # Return the word associated to the predicted index
  return tokenizer.index_word[pred]
################################################################################

