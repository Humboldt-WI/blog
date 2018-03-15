+++
title = "Financial Time Series Predicting with Long Short-Term Memory"
date = '2018-03-15'
tags = [ "Deep Learning", "Time Series", "LSTM", "Long Short Term Memory", "Class17/18", "Share Price Prediction", "Time Series Forecasting" ]
categories = ["seminar"]
banner = "img/seminar/financial_time_series/timeseries.PNG"
author = "Class of Winter Term 2017 / 2018"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = "Prediction of financial time series using LSTM networks "
+++

## Getting started
Failing to forecast the weather can get us wet in the rain, failing to predict stock prices can cause a loss of money and so can an incorrect prediction of a patient’s medical condition lead to health impairments or to decease. However, relying on multiple information sources, using powerful machines and complex algorithms brought us to a point where the prediction error is as little as it has ever been before.

In this blog, we are going to demystify the state-of-the-art technique for predicting financial time series: a neural network called Long Short-Term Memory (LSTM). 

Since every new deep learning problem requires a different treatment, this tutorial begins with a simple 1-layer setup in Keras. Then, in a step-by-step approach we explain the most important parameters of LSTM that are available for model fine-tuning. In the end, we present a more comprehensive multivariate showcase of a prediction problem which adopts Google trends as a second source of information, autoencoders and stacked LSTM layers built to predict share price returns of a major German listed company.

Since this blog post is designed to introduce the reader to the implementation of LSTMs in Keras, we will not go into mathematical and theoretical details here. We can highly recommend and suggest the excellent blogposts by Colah (http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and  (more advanced) Kapur (https://ayearofai.com/rohan-lenny-3-recurrent-neural-networks-10300100899b) to pick up the theoretical framework on LSTMs in general before going through this blogpost. 

To bridge the theoretical gap, we just want to briefly explain why LSTMs should be applied for time series prediction as state-of-the-art machine learning technique. 

## Short Intuition on LSTM architecture
Proposed by Hochreiter and Schmidhuber in 1997, LSTMs provide a solution to the so-called vanishing gradient problem faced by standard recurrent neural networks (RNN). LSTMs are a type of RNNs, so they have the same basic structure. However, the extension of LSTMs is that the LSTM-cell itself (which is part of a recurrent neural network) is a much more extensive series of matrix operations. Ultimately, this advanced version allows the model to learn long-term dependencies by introducing a new kind of state, the LSTM cell state. Instead of computing each hidden state as a direct function of only inputs and other hidden states, LSTMs compute it as a function of the cell state’s value at that time step. The following illustration gives a graphical explanation of LSTMs.


<img align="center" width="80%" style="display:block;margin:0 auto;" 
src="/blog/img/seminar/financial_time_series/LSTM3-chain_orig.png">

The cell state is the top line in the LSTM cell illustrated in the figure above. It can be intuitively thought of as being a conveyor belt which carries long-term memory. Mathematically, it is just a vector. The reason to use this analogy is because information can flow through a cell very easily without the need for the cell state being modified at all. With RNNs, each hidden state takes all the information from before and fully transforms it by applying a function over it. Each component of the hidden state is modified according to the new information at each single time step. In contrast, the LSTM cell state takes information and only selectively modifies it while the information flows through. This methodology solves the vanishing gradient problem. Why? The key is that new information is added, i.e. not multiplied, to the cell state. Different to multiplication in RNNs, addition distributes gradients equally, the chain-rule does not apply. Thus, when we inject a gradient at the end, it will easily flow back all the way to the beginning without the problem to vanish. But enough theory, let’s get our hands dirty with the implementation in Keras.

## Data Collection

