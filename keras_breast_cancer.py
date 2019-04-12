# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:22:14 2019

https://www.kaggle.com/thebrownviking20/intro-to-keras-with-breast-cancer-data-ann

https://www.kaggle.com/thebrownviking20/intro-to-keras-with-breast-cancer-data-ann
@author: vdokku
"""

# Importing libraries
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
# Importing data
data = pd.read_csv('input/data.csv')
# If you want to delete the column. 
del data['Unnamed: 32']

# Show the first 10 rows. 
data.head()


data.shape

"""

data.shape ==> There are 569 ROWS and 32 columns. 
Out[15]: (569, 32)
"""

data.describe

sample_df = data.iloc[:2, 2:]
sample_df.shape
"""
(2, 30)
"""

sample_df_1 = data.iloc[:2, :2]
sample_df_1.shape
"""
(2, 2)
"""

sample_df_2 = data.iloc[:2, 1]
sample_df_2.shape
"""
 (2,)
"""


sample_df_3 = data.iloc[:, 2:]
sample_df_3.shape
"""
 (569, 30)
"""

"""
See the above difference about the colon changes. 

It depends on the position of the colon ;
"""


X  = data.iloc[:, 2:].values # : means all the rows, 2: means staring from 2th index and from there all the full set of columns. 
y = data.iloc[:, 1].values  # All the rows and empty columns


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

"""

LabelEncoder ==> Corresponds to an algorithm where, it needs the data to be numerical format. 

"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

"""
This is one of the model selection API.
"""


"""
This is for scaling the features. 
Features are the attributes of the data
Label is one of the outcomes. 
So What does encoding mean ? 
Why it is names as the LabelEncoder ?? 
"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




"""
What is the meaning of the Feature Scaling ? 
What is the meaning of the Feature Engineering ? 
What is the difference ?? 
"""
"""

Enough of the data... Let's import the KERAS and start working on it. 
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))

"""
Overfitting valla our model will not be able to 
get the right amount of data. 
"""

"""

input_dim - number of columns of the 
dataset

output_dim - number of outputs to be 
fed to the next layer, if any

activation - activation function which is ReLU in this case

init - the way in which weights 
should be provided to an ANN

The ReLU function is f(x)=max(0,x). 
Usually this is applied element-wise 
to the output of some other function, 
such as a matrix-vector product. 

In MLP usages, rectifier units 
replace all other activation 
functions except perhaps the 
readout layer. But I suppose you 
could mix-and-match them if you'd 
like. 

One way ReLUs improve neural networks 
is by speeding up training. 
The gradient computation is very 
simple (either 0 or 1 depending on 
the sign of x). 

Also, the computational step of a 
ReLU is easy: any negative elements 
are set to 0.0 -- no exponentials, 
no multiplication or division 
operations. Gradients of logistic 
and hyperbolic tangent networks are 
smaller than the positive portion of 
the ReLU. This means that the 
positive portion is updated more 
rapidly as training progresses. 

However, this comes at a cost. 

The 0 gradient on the left-hand 
side is has its own problem, called 
"dead neurons," in which a gradient 
update sets the incoming values to a 
ReLU such that the output is always 
zero; modified ReLU units such as 
ELU (or Leaky ReLU etc.) 
can minimize this. 


"""

# Adding the second hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


# Activation ==> RELU & SIGMOID.



"""

output_dim is 1 as we want only 1 
output from the final layer.

Sigmoid function is used when 
dealing with classfication 
problems with 2 types of results.
(Submax function is used for 3 or 
more classification results)


"""

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""


Optimizer is chosen as adam for 
gradient descent.

Binary_crossentropy is the loss function used.

Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0. More about this


"""

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=150)
# Long scroll ahead but worth
# The batch size and number of epochs have been set using trial and error. Still looking for more efficient ways. Open to suggestions. 

"""



Batch size defines number of samples that going to be propagated through the network.

An Epoch is a complete pass through all the training data.


"""
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))


sns.heatmap(cm,annot=True)
plt.savefig('h.png')


























