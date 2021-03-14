# Natural_Language_Processing

Amazon UCI Sentiment Analysis Using Neural Networks

Research Question

To what extent of accuracy can the sentiment’s positivity (or negativity) be predicted and what can this do for strategic management initiatives? How can the model further company insight into customer disposition and help review sentiment analysis?
  
Objectives and Goals

Using python packages, Tensorflow and Keras, a LSTM model will be built to predict the disposition of the sentiment recorded via Amazon reviews. The main objective being to answer the research question. Certain goals of the project are to preprocess the data for optimal model performance, adhere to Keras/Tensorflow assumptions, and to provide an 80% or higher accuracy metric with a 90% AUC or higher. Lastly, to gather the three most used words within the positive and negative groups to provide the company with actionable items to focus on in production, distribution, and marketing. 
  
Prescribed Network

The preferrable type of neural network that can be modeled and trained for this type of bimodal classification is the LSTM Keras deep learning model. Within its functionality, it offers the most control of its hyperparameters. Some assumptions of this network are datatype, tokenized text, underlying patterns and distributions within the data, and standardized, sequential data.
  
Data Exploration

The data, prior to tokenization and fitting of a model, needs to be imported, cleaned, and explored. This was accomplished by exploring the presence of unusual characters, vocabulary size, word embedded text length, and statistical justifications for max sequence length.
  
Vocabulary Size

Broken down, the average text length for each of the reviews is 55.226 characters. The maximum review text length is 149 characters. The total amount of characters accounted for throughout all reviews is 55,226. The total unique characters appearing throughout the reviews is 83. This exploration will allow for a better understanding of realistic hyperparameters for our model.
Proposed Word Embedding Length
Word embedding is the act of transforming text into a communicable variable for a machine learning process. A perfect example of this is one hot encoding. The tokenization process is the preprocessing requirement for embedding. Further, the Keras LSTM also contains a hyperparameter which can have a range of embedding specifications made for preferrable model output. Embedding length, per industry standard, is usually no larger than 50. This number is also smaller with smaller datasets containing simple character variances. The best number for embedded length will be determined via a for loop which will iterate through epochs, batch sizes, embedded lengths, sequence lengths and learning rates. The iterations will explore embedding lengths of 1 through 32.
The result of the iteration shows the most effective embedding length of 3. The screenshot of the code and output is provided below.

Statistical Justification for Max Sequence Length

Per the code outputs above, it is shown that the max tokenized sequence length is 28 with 75% of the data falling below a length of 15 and 50% falling below 9. This means the data is skewed right lying heavily towards a shorter sequence length. The model will be most accurate in its predictions with a lower max sequence length. Analogous to the process for determining the max embedded length, hyperparameters will be iterated through to find the most accurate model with the least loss. The results of the iterative process are below with a max sequence length of 19.
  
Tokenization

Tokenizing textual data is preprocessing in nature as it transforms the data into a useable form for a machine learning model. The LSTM Keras model assumes numerical input hence the need for transformation. The packages used for this process are Tensorflow’s Tokenizer and pad_sequences. These packages are used to symbolicaly represent the textual data and normalize it. This makes the data useable by a machine learning model.  Here is a screenshot of the code used and an example of tokenized textual data.

Padding Process

One of the assumptions of a LSTM model in Keras is that the sequence variance is standardized. This means that the data fed to the model must be non-variable or equal in length. Masking and padding in Keras allow for this requirement to be met. Truncating and padding is the parameter within the function for cutting off or adding on lengths to the sequence to guarantee this assumption. The data’s sequence distribution lies heavily in the lower range, so these are the areas to be padded and truncated. Below is a screenshot of the code used and an example of the padding.

Categories of Sentiment

There are two categories of sentiment, positive and negative. This in turn provides the need for binary classification activation function. The combination of the ‘relu’ activation function and the ‘sigmoid’ activation functions will allow for non-linear and linear forward propagation through the neural network. This combination is extremely powerful in machine learning and will provide the most accurate prediction outcomes for this bimodal classification.

Steps to Prepare the Data

The training and test split used was 20% of the original dataset. This is industry standard for sampling as it more accurately represents the whole of the data. The validation and test split were 20% of the training test split. This allows for micro validation throughout the model. Their size is shown in the below screenshot. 

The steps for preparing the data as well as the steps for neural networks and NLP techniques are provided below.

1.	Import packages necessary for the project: Numpy, Pandas, Tensorflow, Keras, Sklearn, Ax.Client, Matplotlib, Seaborn, Base64, and Ipython.display for HTML.
2.	Read in the text file from UCI and fit to dataframe. Parse out by start and end tokens to ensure separation of reviews and their accompanying sentiment attachment.
3.	Use data exploration techniques for cleaning, preprocessing and data transformation.
a.	Display data-frame shape.
b.	Display null values.
c.	Display bimodal sentiment counts.
d.	Graph bimodal sentiment counts.
e.	Display a random row for accuracy in parsing reviews.
f.	Discover average, max, total characters, and unique characters throughout reviews.
4.	Perform initial train test split of the dataset with a 20% sampling size (industry standard).
5.	Perform initial train1 validation split of the train test split with a 20% sampling size.
6.	Parse characters for character distribution throughout dataset
7.	Tokenize the dataset for vocabulary size of 55,226 (this allows for each textual character to be represented numerically within the model).
8.	Display a distribution visualization and statistic for the tokenized textual data. 
9.	Build a function for Keras model hyperparameter manipulation.
10.	Build a variable for a range of tests for each hyperparameter e.g., learning rate, dropout rate, LSTM units, dense neurons, number of epochs, batch size, embedding size, and max text length.
11.	Use Ax client server to iterate over all possible hyperparameter manipulations throughout the range of the variable specified in step 10.
12.	Print the best parameters from the test completed in step 11.
13.	Fit the model with the best parameters determined in step 12.
14.	Pre-pad and pre-truncate the sequences for standardized model input. This will avoid overfitting of the model.
15.	Evaluate the model with an AUC score.
16.	Plot a visualization of the distribution of the predictions.
17.	Download the fully prepared dataset.

Model Summary and Network Architecture

The model summary from Tensorflow shows the different layers and parameters and their corresponding metrics. For this model, the input layer has a parameter number of 0, and an output shape of (,113). The embedding has an output shape of(,113), and a parameter number of 552,260. The LSTM has an output shape of (,5) and a parameter number of 320. The dense layer has an output shape of (,69) and a parameter number of 414. The dropout has an output shape of (,69) and a parameter number of 0. The dense_1 layer has a parameter number of 70 and an output shape of (,1).  The total number of parameters is 553,064. The output of this model summary is shown below as is the model architecture visualization.

Hyperparameters
	
The choice of hyperparameters is justified by the model’s accuracy and AUC alone. However, there is further justification to be shown in each hyperparameter by the exploration done via the iteration over parameter manipulation with ax_client. Below, each of the following hyperparameters will be justified: activation functions, number of nodes per layer, loss function, optimizer, stopping criteria, and evaluation metric.
  
Activation Functions

Those activation functions used in the model are namely ‘relu’ and ‘sigmoid’. The linear and non-linearity incorporated in the ‘relu’ activation function makes the forward propagation through neural networks extremely accurate. Comparing to ‘tanh’, ‘relu’ does not clip at a certain y disallowing for neural activity beyond certain weight ranges. Lastly, ‘relu’ is an industry standard activation function simply because of its versatility and accuracy. 

Sigmoid was used with a bimodal categorical cross entropy loss function. This allowed for binary classification without the limitations of node death from the ‘softmax’ function. Sigmoid, like ‘relu’, allows for linear and non-linearities in node weight evaluation and forward propagation. 

These two together are what allowed for a model AUC score of 97.43%.

Number of Nodes per Layer

The number of nodes per layer was determined by the iterative process provided by ax_client. This iterated through a range for each hyperparameter and the following code was used to determine the best number of nodes per layer. Below is the screenshot.



Loss Function

The loss function was chosen because of the nature of the binary classification problem for sentiment analysis. Bimodal categorical cross entropy was used for model optimization. Had the ‘softmax’ activation function been used, then a simple cross entropy loss function would have been used. However, given the complex forward propagation throughout the neural network, something fitting of a binary classification and linear propagation was needed. This meant the use of non-standard means such as the ‘sigmoid’ activation function. Because this activation function was used, the binary cross entropy loss function was used in tandem. 
  
Optimizer

‘Adam’ was the optimizer used for this hyperparameter. Adaptive Moment Estimation is the full name and meaning of this optimizer. It allows for in the moment estimations and adjustments using the learning rate provided. This prevents overfitting and makes the optimizer preferable to standard gradient descent or RMSprop or momentum optimizers.
  
Stopping Criteria

The stopping criteria used for the model is a range for each hyperparameter. In short, each parameter was given a range of test values. Then ax_client iterated through each range for the possible model manipulations; thus, it was able to ensure the perfect model hyperparameters were selected. The ‘stopping criteria’ or range of test parameters for each hyper parameter is in the table below.

Hyperparameter	Range of Stopping Criteria
Learning Rate	[0.0001, 0.001]
Dropout Rate	[0.01. 0.02]
LSTM Units	[5, 10]
Dense Neurons	[1, 150]
Number of Epochs	[10, 20]
Batch Size	[10, 20]
Embedding Size	[2, 30]
Max Text Length	[10, 150]

Evaluation Metric

The evaluation metric used for the model was area under curve (AUC) and ‘accuracy’ within the epoch iteration. The higher the AUC, the more your model is accounting for the data. The accuracy of the epoch shows the accuracy of the training to validation e.g., actual to prediction outputs. Both are commonplace and industry standard when working in Keras.
  
Stopping Criteria

Stopping criteria in Keras has inputs of patience, val_loss, min_delta, verbose, and mode. This allows for the model to self-evaluate its epochs at the completion of each epoch. This tells the model to either continue or stop once a certain criterion is met. The value in this method is when new data is added to the dataset, the stopping criteria can stop at a predetermined desirability for hyperparameters. It can also be viewed as an automation technique. The following screenshot is provided to show the last training epoch output.

Fit

There are several measures of the fitness of a model. These are: accuracy, loss, ROC/AUC, and model evaluation. The model formed in this analysis has an accuracy of 0.9969, and an AUC of 0.975474. These prove an extremely fit model. Furthermore, the combination of linear and non-linear activation functions addresses overfitting. Underfitting and overfitting was addressed by the sample size as well as the cross-evaluations performed in the model compilation.
  
Predictive Accuracy

The evaluation metric selected from the prior section was accuracy and AUC/ROC. These two scores provide insight into model useability and reliability. It should be noted that both the accuracy score and AUC score determined are only in terms of data within the dataset. Therefore, it is not determined to be reliable with new data. However, the way the model has been built and the preprocessing transformations have enabled the model to be adaptable to new data. The accuracy and AUC will need to be assessed with any addition of new data. As it stands, the accuracy of the model’s predictive abilities is 99.69% and its AUC score is .975474. Both are extremely high which is an incredibly good sign!  
 
Functionality
	The neural network uses activation function ‘relu’ and ‘sigmoid’. This is a combination of linear and non-linear forward propagation. The functionality in this is a model that leans more towards underfitting and node death rather than overfitting. The network architecture is as follows:

Recommendations

The findings of the analysis provide a model of 99.69% accuracy and an AUC of .975474. This has led to the discovery of the exact character distributions that help to classify sentiments as positive or negative. The model measurements of fitness have proven its accuracy. However, the limitations of this model should be noted as it has not been tested against new data. As new data is gathered, the model should be reassessed for fitness. There are certain autonomic processes built into the project for the ease of adding new data. Those processes automated are tokenization, train/test splits, and hyperparameter testing and assignment.  

These limitations acknowledged, immediate findings which may be produced are the prediction of sentiment based off characters contained in a textual review. This allows the company to use this model to focus marketing initiatives on those words and phrases which appear most often within the reviews. Furthermore, manufacturing, research and development, and operations can look at those words and phrases most associated with negative reviews so products may be improved.

The recommendation is this, to use the model to discover reasons for customer dispositions within their reviews and address these concerns with strategic management initiatives.
  
Reporting
The official report of the analysis will be submitted as an ipynb file.

Sources for Third Party Code
	The below website was used to guide the intense process of this project. The code was followed, not copied. 
How to do Sentiment Analysis with Deep Learning (LSTM Keras) - Just into Data
 

References

Kotzias, D. (2015). From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015 [Scholarly project]. In UCI Machine Learning Repository. Retrieved February 28, 2021, from https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
Justin, L. (2020, March 29). How to do sentiment analysis with deep Learning (LSTM KERAS). Retrieved March 01, 2021, from https://www.justintodata.com/sentiment-analysis-with-deep-learning-lstm-keras-python/


