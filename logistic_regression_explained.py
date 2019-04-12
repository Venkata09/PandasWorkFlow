# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:09:29 2019

@author: vdokku
"""

hoursStudied=[
    [1.0],
    [1.5],
    [2.0],
    [2.5],
    [3.0],
    [3.5],
    [3.6],
    [4.2],
    [4.5],
    [5.4],
    [6.8],
    [6.9],
    [7.2],
    [7.4],
    [8.1],
    [8.2],
    [8.5],
    [9.4],
    [9.5],
    [10.2]]

passed =     [0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1]


print("hoursStudied  passed")

# You are trying to zip the two lists 
for row in zip(hoursStudied, passed):
    print("  ",row[0][0],"    ----->",row[1])
    
    
    
import matplotlib.pyplot as plt
plt.scatter(hoursStudied, passed, color='black')
plt.xlabel("Hours Studied")
plt.ylabel("Passed")

# If we plot a normal linear regression over our data points, it looks like this:

import math
import matplotlib.pyplot as plt
import numpy as np

""" Now let's define the sigmoid function. 

f(x) = 1/1 + e power -x.
"""


def sigmoid(x):
    new_array = []
    for each_item_in_x in x:
        new_array.append(1/1+math.exp(each_item_in_x))
    return new_array

# Now, we'll generate some values for x. 
# It will have values from -10 to +10 with increments of 0.2.
#              Lower Limit,Upper Limit, How much of an increment.
x = np.arange(-10.,           10.,            0.2)

return_list = sigmoid(x)

for each_entry in return_list:
    print(each_entry)


plt.plot(x,return_list)
plt.show()






