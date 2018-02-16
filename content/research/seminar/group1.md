+++
title = "Neural Network Fundamentals II"
date = '2018-02-01'
tags = [ "Deep Learning", "Neural Networks", "Class17/18","Blog Instruction"]
categories = ["seminar"]
description = "Introduction to the Neural Network Fundementals"
banner = "img/banners/seminar_nn.png"
author = "Group NN"
disqusShortname = "https-wisample-github-io-blog"
+++

A Gentle Introduction to Neural Network Fundamentals
=============

<script src="https://cdn.datacamp.com/datacamp-light-latest.min.js"></script>


### Implementation of the NN from scratch
Let's try to reimlement such a structure using Python.
The crutial elements are:

* layers
* nodes
* weights between them
* activation function




<br/>

 
<div data-datacamp-exercise data-lang="python">
<code data-type="pre-exercise-code">

</code>
<code data-type="sample-code">
import numpy as np	
		
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
	
import matplotlib.pyplot as plt

 # Draw this function
x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x))
plt.show()
</code>

<code data-type="solution">
import numpy as np	
		
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
	
import matplotlib.pyplot as plt

# Draw this function
x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x))
plt.show()
</code>

<code data-type="sct">

success_msg("Great job!")
</code>
<div data-type="hint">Use the assignment operator (<code>=</code>) to create the variable <code>a</code>.</div>
</div>



  
<div data-datacamp-exercise data-lang="python">
<code data-type="pre-exercise-code">
  # no pec
</code>
<code data-type="sample-code">
  # Create a variable a, equal to 5
  a = 5

  # Print out a
  print(a)

</code>
<code data-type="solution">
  # Create a variable a, equal to 5
  a = 5

  # Print out a
  print(a)
</code>
<code data-type="sct">
  test_object("a")
  test_function("print")
  success_msg("Great job!")
</code>
<div data-type="hint">Use the assignment operator (<code>=</code>) to create the variable <code>a</code>.</div>
</div>







# Copy and Paste Example from Group 1




<div data-datacamp-exercise data-lang="python">
<code data-type="pre-exercise-code">
# Load the package to work with numbers
import numpy as np


# In[50]:

# Determine the structure of the NN
i_n = 3
h_n = 5
o_n = 2
# i_n, h_n and o_n stand here for the number of nodes in input, hidden and output layers respectively 
# - exactly as on the picture above


# In[52]:

# Randomly define the weights between nodes 
w_i_h = h_n, i_n
w_h_o = o_n, h_n

# Show matrices of randomly assigned weights
w_i_h
w_h_o


# ## Activation Function

# Besides complicated multilayer structure with many nodes neurosystems in nature has one more important feature - neurons in them send signal further or "fire" only when they get a signal that is strong enough - stronger than certain treshold. This can be represented by a step function.

# <img src="pics/step_function.png" alt="Drawing" style="width: 700px;"/> [Source: https://www.researchgate.net/figure/Three-different-types-of-transfer-function-step-sigmoid-and-linear-in-unipolar-and_306323136]

# In[53]:

# Determine activation function which is an approximation for "firing" of neurons
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[54]:

# Draw this function
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x))
plt.show()
</code>
<code data-type="sample-code">
 
# Load the package to work with numbers
import numpy as np


# In[50]:

# Determine the structure of the NN
i_n = 3
h_n = 5
o_n = 2
# i_n, h_n and o_n stand here for the number of nodes in input, hidden and output layers respectively 
# - exactly as on the picture above


# In[52]:

# Randomly define the weights between nodes 
w_i_h = h_n, i_n
w_h_o = o_n, h_n

# Show matrices of randomly assigned weights
w_i_h
# w_h_o


# ## Activation Function

# Besides complicated multilayer structure with many nodes neurosystems in nature has one more important feature - neurons in them send signal further or "fire" only when they get a signal that is strong enough - stronger than certain treshold. This can be represented by a step function.

# <img src="pics/step_function.png" alt="Drawing" style="width: 700px;"/> [Source: https://www.researchgate.net/figure/Three-different-types-of-transfer-function-step-sigmoid-and-linear-in-unipolar-and_306323136]

# In[53]:

# Determine activation function which is an approximation for "firing" of neurons
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[54]:

# Draw this function
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x))
plt.show()
 
 
</code>

<code data-type="solution">
 # Load the package to work with numbers
import numpy as np


# In[50]:

# Determine the structure of the NN
i_n = 3
h_n = 5
o_n = 2
# i_n, h_n and o_n stand here for the number of nodes in input, hidden and output layers respectively 
# - exactly as on the picture above


# In[52]:

# Randomly define the weights between nodes 
w_i_h = h_n, i_n
w_h_o = o_n, h_n

# Show matrices of randomly assigned weights
w_i_h
# w_h_o


# ## Activation Function

# Besides complicated multilayer structure with many nodes neurosystems in nature has one more important feature - neurons in them send signal further or "fire" only when they get a signal that is strong enough - stronger than certain treshold. This can be represented by a step function.

# <img src="pics/step_function.png" alt="Drawing" style="width: 700px;"/> [Source: https://www.researchgate.net/figure/Three-different-types-of-transfer-function-step-sigmoid-and-linear-in-unipolar-and_306323136]

# In[53]:

# Determine activation function which is an approximation for "firing" of neurons
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[54]:

# Draw this function
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x))
plt.show()
 
 
</code>

<code data-type="sct">

success_msg("Great job!")
</code>

<div data-type="hint">Use the assignment operator (<code>=</code>) to create the variable <code>a</code>.</div>
</div>