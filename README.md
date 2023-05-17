# MLP---1
Stock Trend Prediction Web App


Industry 
Finance | Stocks | Investments 

Skills
Python | Streamlit | Web scraping | API | Deployment | Deep learning | Machine Learning

Problem statements
Build a web application that predicts stocks.

Data Structure 
Data was scraped from yahoo finance API. The Yahoo Finance API provides access to a vast dataset of historical stock prices and financial information for publicly traded companies. This dataset contains information on a wide range of stocks, including stocks from major indices such as the S&P 500 and NASDAQ, as well as smaller, less well-known companies. The data includes a variety of financial metrics, including opening and closing prices, daily high and low prices, trading volume, and more, making it a valuable resource for investors, analysts, and researchers.
The dataset contains historical stock data for all the stocks in Yahoo Finance API, the files are named after their respective stock symbols. Each file is in CSV format and will contain the following columns:
Date: The date of the trading day.
Open: The opening price of the stock for that day.
Close: The closing price of the stock for that day.
High: The highest price the stock reached during that day.
Low: The lowest price the stock reached during that day.
Volume: The number of shares of that stock that were traded during that day.
Dividend: Any dividends paid out on that day.
Stock Split: Any stock splits that occurred on that day.
Each row in the file represents one day of trading for that stock.
Methods
A. EDA and data manipulation
Inspect columns and shape of dataframe
Drop columns
Creating functions  moving averages ma100 and ma200
Plot moving averages with stock

B. Preprocessing
Split dataset into data training (70% of dataset) and data testing (30% of dataset)
Create a scaler function for normalizing the data
Apply normalization process to data training set




C. Training data creation
PART 1

Create training data for machine learning model
Create sliding window of length 100 over the data_training_array, where each window is used as an input to the model and the corresponding value after the window (i.e., the 101st element and beyond) is used as the target output.
Assuming that data_training_array is a one-dimensional array of numerical values, here's a corrected version of the code that should work. 
x_train is a list of numpy arrays of shape (100,) and y_train is a list of individual values.
 Each element of x_train is a sliding window of length 100 over data_training_array, and the corresponding element of y_train is the value immediately after the end of the window.
Convert to  x_train  and y_train np array


D.  Build the model

The code model = Sequential() creates a new instance of a Keras Sequential model. In this case, we will be adding an LSTM layer followed by a Dropout layer to this model.
model.add… adds a new layer to the Sequential model using the add method. Specifically, we're adding an LSTM layer with the following parameters:
 50 LSTM units in the layer. 
ReLU (Rectified Linear Unit ) activation function.ReLU function returns 0 if the input is negative, and returns the input value directly if it is positive. An activation function is a mathematical function that is applied to the output of each neuron in a neural network. The purpose of the activation function is to introduce non-linearity into the output of the neuron, which allows the neural network to learn complex, non-linear relationships in the data.The activation function is then applied to the weighted sum to produce the output of the neuron.
return_sequences:stack multiple LSTM layers, we need to set this parameter to True so that we get a sequence of outputs that can be fed into the next LSTM layer. 
input_shape: The shape of the input data. 
The Dropout helps to prevent overfitting. In this case, we're specifying a dropout rate of 0.2, which means that 20% of the input units will be randomly set to 0 at each training update.Overfitting is a common problem in machine learning where a model is trained too well on the training data, and as a result, it performs poorly on new, unseen data.
In overfitting, the model learns the noise or random fluctuations in the training data rather than the underlying patterns or trends. This causes the model to fit the training data too closely, to the point that it memorizes the training data instead of generalizing to new data.
Together, these three lines of code create a Keras Sequential model with a single LSTM layer, with 50 units and ReLU activation, that takes input sequences of length 100 and returns output sequences of the same length. The Dropout layer is added after the LSTM layer to help prevent overfitting.
The code block is defining a sequential neural network model using the Keras library in Python. The model architecture consists of four LSTM (Long Short-Term Memory) layers followed by a dense output layer.

Each LSTM layer has a different number of units, or neurons, and uses the ReLU activation function. The return_sequences parameter is set to True for the first three LSTM layers, which means that the output of each layer is fed as input to the next layer in the sequence.

Additionally, a Dropout layer is added after each LSTM layer, with different dropout rates for each layer. The purpose of the Dropout layer is to randomly drop out (set to zero) a fraction of the inputs to the layer during training, which can help prevent overfitting by introducing some noise into the network and encouraging it to learn more robust features.

Finally, a Dense output layer is added with a single unit, which will output a single scalar prediction. The purpose of the output layer is to map the features learned by the LSTM layers to the target variable, in this case, a time series prediction.

In summary, the code block is defining a deep LSTM model with Dropout regularization and a Dense output layer, which can be used for time series prediction tasks.

Then the model is compiled and trained  

Adam optimizer is used, which is a popular optimization algorithm for training deep neural networks. The loss function is set to mean squared error (MSE), which is a commonly used loss function for regression problems.In machine learning, an optimizer is an algorithm that adjusts the parameters of a model in order to minimize its loss function. The loss function measures the difference between the predicted output and the true output for a given input.The goal of an optimizer is to find the values of the model's parameters that minimize the loss function, which corresponds to making the model's predictions as accurate as possible. This is done by iteratively updating the model's parameters based on the gradients of the loss function with respect to the parameters, using a variant of stochastic gradient descent (SGD) or another optimization algorithm.There are many different types of optimizers available in machine learning, each with their own advantages and disadvantages. Some common optimizers include stochastic gradient descent (SGD), Adam, RMSprop, and Adagrad. The choice of optimizer can have a significant impact on the performance of a model, and it often depends on the specific characteristics of the problem being solved.

The fit() method trains the model on the input data (x_train) and the corresponding target output (y_train) for a specified number of epochs. In this case, the model is trained for 50 epochs. During training, the model tries to minimize the MSE loss between the predicted output and the true output. The training process involves iteratively updating the weights and biases of the model to minimize the loss using backpropagation.In machine learning, loss refers to the error or cost associated with a model's predictions. It measures the difference between the predicted output and the true output for a given input. The goal of a machine learning algorithm is to minimize the loss function by adjusting the model's parameters, so that it can make more accurate predictions on new, unseen data.There are many different types of loss functions in machine learning, depending on the type of problem being solved. For example, for regression problems, the most commonly used loss function is mean squared error (MSE), which measures the average squared difference between the predicted output and the true output. For classification problems, cross-entropy loss is commonly used, which measures the difference between the predicted probability distribution and the true probability distribution of the classes. Choosing an appropriate loss function is important in machine learning, as it can have a significant impact on the performance of the model. The choice of loss function depends on the specific problem being solved, as well as other factors such as the type of data, the amount of training data available, and the complexity of the model.

After training, the model will be able to predict the target variable for new input data with some level of accuracy, depending on the complexity of the problem and the quality of the input data.
In machine learning, an epoch refers to one complete iteration of the training dataset through the neural network during the training phase.
In each epoch, the neural network receives the entire training set, processes it forward through the network to make predictions, and then calculates the loss between the predicted output and the true output for each example in the training set. It then updates the weights and biases in the network based on the gradients of the loss function with respect to the parameters, using an optimizer such as stochastic gradient descent (SGD).
Training a neural network typically involves multiple epochs, where the same training data is passed through the network repeatedly until the desired level of performance is achieved. The number of epochs is a hyperparameter that must be set by the user, and it can have a significant impact on the performance of the model. Setting the number of epochs too low may result in underfitting, while setting it too high may result in overfitting.
model.save('keras_model.h5') saves the trained Keras model to a file named 'keras_model.h5' in the current directory. The saved model can be loaded later using keras.models.load_model() function and used for prediction or further training.

E. Testing model

The  scaler is being used to scale the numerical values in the final_df DataFrame.

The fit_transform() method of the StandardScaler object is being called on final_df. This method first fits the scaler to the data by computing the mean and variance of the data, and then transforms the data by subtracting the mean and dividing by the standard deviation.

The scaled data is then returned and assigned to a new variable called input_data. The input_data variable now contains a NumPy array with scaled values of the numerical columns in final_df.

Scaling the input data is a common data preprocessing technique used in machine learning to normalize the features and improve the performance of the learning algorithms.

The  input_data NumPy array that was created by scaling the values of final_df is being used to create the test data for a machine learning model.

The x_test and y_test lists are initialized as empty lists. These lists will later be populated with the input features and output values for the test set.

The for loop iterates over the rows of the input_data array starting from index 100, since the first 100 rows of the array are used to create the first input sequence.

For each row i in input_data after the 100th row, a new input sequence is created by selecting the previous 100 rows, starting from i-100 and ending at i. These 100 rows are then appended to the x_test list.

The corresponding output value for each input sequence is the first column of the input_data array for the row at index i. These output values are appended to the y_test list.

Finally, the x_test and y_test lists are converted to NumPy arrays using np.array(). The shapes of the resulting arrays are printed to confirm that the input and output arrays have the expected dimensions for training a machine learning model.

The y predicted object represents  a trained machine learning model called model is being used to predict the output values for the test input data x_test. The predicted output values are then assigned to a new variable called y_predicted.

The y_predicted variable now contains a NumPy array with the predicted output values for the test input data. These predicted output values can be compared with the actual output values in the y_test array to evaluate the performance of the machine learning model.

The scale_ attribute of a trained Scikit-learn StandardScaler object is being accessed to obtain the scaling factors used to scale the input data.The scale_ attribute is an array that contains the standard deviation of each feature in the input data. Each feature is scaled by dividing it by its corresponding standard deviation.

By accessing the scale_ attribute, the scaling factors used to transform the input data into a normalized form can be retrieved. These scaling factors can be useful for performing inverse transformations to rescale the output of a machine learning model to the original scale of the data.

The predicted and actual output values for the test data are being rescaled back to their original units using a scale factor.

The scale_factor is being calculated as the reciprocal of the scaling factor that was used to scale the input data using the StandardScaler object. Specifically, the scale_factor is calculated as 1/0.2099517, where 0.2099517 is the standard deviation of the first column of the input data that was used to fit the StandardScaler object.

The predicted output values in y_predicted and actual output values in y_test are then multiplied by the scale_factor. This rescales the output values back to their original units before they were normalized.

The purpose of this rescaling is to make the predicted and actual output values directly comparable to the original, unscaled output values. This allows for a more meaningful evaluation of the performance of the machine learning model on the test data.

The program was deployed via streamlit 

