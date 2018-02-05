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


