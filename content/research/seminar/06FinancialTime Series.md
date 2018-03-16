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

## Introduction
Failing to forecast the weather can get us wet in the rain, failing to predict stock prices can cause a loss of money and so can an incorrect prediction of a patient’s medical condition lead to health impairments or to decease. However, relying on multiple information sources, using powerful machines and complex algorithms brought us to a point where the prediction error is as little as it has ever been before.

In this blog, we are going to demystify the state-of-the-art technique for predicting financial time series: a neural network called Long Short-Term Memory (LSTM). 

Since every new deep learning problem requires a different treatment, this tutorial begins with a simple 1-layer setup in Keras. Then, in a step-by-step approach we explain the most important parameters of LSTM that are available for model fine-tuning. In the end, we present a more comprehensive multivariate showcase of a prediction problem which adopts Google trends as a second source of information, autoencoders and stacked LSTM layers built to predict share price returns of a major German listed company.

Since this blog post is designed to introduce the reader to the implementation of LSTMs in Keras, we will not go into mathematical and theoretical details here. We can highly recommend and suggest the excellent blogposts by <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">Colah</a> and  <a href="https://ayearofai.com/rohan-lenny-3-recurrent-neural-networks-10300100899b">Kapur (more advanced)</a> to pick up the theoretical framework on LSTMs in general before going through this blogpost. 

To bridge the theoretical gap, we just want to briefly explain why LSTMs should be applied for time series prediction as state-of-the-art machine learning technique. 

## Short Intuition on LSTM architecture
Proposed by Hochreiter and Schmidhuber in 1997, LSTMs provide a solution to the so-called vanishing gradient problem faced by standard recurrent neural networks (RNN). LSTMs are a type of RNNs, so they have the same basic structure. However, the extension of LSTMs is that the LSTM-cell itself (which is part of a recurrent neural network) is a much more extensive series of matrix operations. Ultimately, this advanced version allows the model to learn long-term dependencies by introducing a new kind of state, the LSTM cell state. Instead of computing each hidden state as a direct function of only inputs and other hidden states, LSTMs compute it as a function of the cell state’s value at that time step. The following illustration gives a graphical explanation of LSTMs.


<img align="center" width="80%" style="display:block;margin:0 auto;" 
src="/blog/img/seminar/financial_time_series/LSTM3-chain_orig.png">
<i style="float:right;">Figure from: <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">Colah's Blog</a></i>
<br><br style="clear: both">
The cell state is the top line in the LSTM cell illustrated in the figure above. It can be intuitively thought of as being a conveyor belt which carries long-term memory. Mathematically, it is just a vector. The reason to use this analogy is because information can flow through a cell very easily without the need for the cell state being modified at all. With RNNs, each hidden state takes all the information from before and fully transforms it by applying a function over it. Each component of the hidden state is modified according to the new information at each single time step. In contrast, the LSTM cell state takes information and only selectively modifies it while the existing information flows through. This methodology solves the vanishing gradient problem. Why? The key is that new information is added, i.e. not multiplied, to the cell state. Different to multiplication in RNNs, addition distributes gradients equally, the chain-rule does not apply. Thus, when we inject a gradient at the end, it will easily flow back all the way to the beginning without the problem to vanish. But enough theory, let’s get our hands dirty with the implementation in Keras.

## Getting started...
### Data Collection 

Before we can start our journey we would like to introduce two useful APIs that can make your life a lot easier:

* `Pandas_datareader` can be used to download finance data via the Yahoo Finance API.
  You can see a small snipped down below. Include your stock and your time frame.
  API does normally not provide returns, which are commonly used in practice. We added another line that calculate log returns for your. Log returns are great! Check it out: Here is a good <a href="https://trends.google.com/trends/explore?q=VW">Google trend API</a>. 

* `Pytrends` can access the <a href="https://medium.com/@pewresearch/using-google-trends-data-for-research-here-are-6-questions-to-ask-a7097f5fb526">reference</a>. In the second box you can find a more comprehensive example that collects data from Google, loops the code, and can even continue downloading at another day (your daily quota is unfortunately limited). Please have a look at the code. If you spend a moment, we are sure you can figure it out. For more information: Here is a good <a href="https://medium.com/@pewresearch/using-google-trends-data-for-research-here-are-6-questions-to-ask-a7097f5fb526">reference</a>. The details of the snipped are not to important for
our modelling, but have a look!

{{< highlight python "style=emacs" >}}
# Get Yahoo Data
import pandas_datareader as pdr

stock = pdr.get_data_yahoo(symbols='#your stock ticker', start=datetime(2012, 1, 1), end=datetime(2017, 12, 31))
stock = stock['Adj Close'] # or one of the other columns (i.e., opening prices, Volumes)
stock_returns = np.log(stock/stock.shift())[1:] #calculate log returns
{{< /highlight >}}



{{< highlight python "style=emacs" >}}
# Download Data from google Trends
from pytrends.request import TrendReq
import datetime
import os
google_username = #(...) #INCLUDE YOUR GOOGLE ACCOUNT
google_password = #(...)
pytrend = TrendReq(google_username, google_password, custom_useragent="My Pytrends Script")

formatter = "{:02d}".format

databegin = list(map(formatter, range(0, 19, 3))) #change TIME here.
dataend   = list(map(formatter, range(4, 25, 3))) #downloads for hourly data atm

lastdate = datetime.date(2018, 3, 12) # Until when do you want to download

file = open("daysleft.txt")
daysprior = int(file.read())
file.close()

while daysprior>2:
    
    file = open("daysleft.txt", "w")
    file.write(str(daysprior))
    file.close()
    
    daysbefore = lastdate - datetime.timedelta(days=daysprior)
    
    keywords = ["Volkswagen"] #INCLUDE KEYWORDS HERE! 
    for i in range(0, len(databegin)):
        begin = daysbefore.strftime("%Y-%m-%d") + "T" + databegin[i]
        end   = daysbefore.strftime("%Y-%m-%d") + "T" + dataend[i]
        
        timeframestring = begin + " " + end
        
        for j in range (0, len(keywords)):
            pytrend.build_payload(kw_list=[keywords[j]], timeframe=timeframestring)
            df = pytrend.interest_over_time()
            df.to_csv("../data/" + keywords[j] + "/" + timeframestring + ".csv")
            
    begin = daysbefore.strftime("%Y-%m-%d") + "T21"
    end   = (lastdate - datetime.timedelta(days=daysprior - 1)).strftime("%Y-%m-%d") + "T01"
    timeframestring = begin + " " + end
    for j in range (0, len(keywords)):
        pytrend.build_payload(kw_list=[keywords[j]], timeframe=timeframestring)
        df = pytrend.interest_over_time()
        df.to_csv("../data/" + keywords[j] + "/" + timeframestring + ".csv")
    
    daysprior = daysprior -1
{{< /highlight >}}

This should just give you an idea on where to start. If you work with financial data, these should come in handy. We are now going to use a dataset that is already prepared to showcase sequence modelling with Keras. The data was collected from the same APIs, you just learnt about. Our dataset has 7 columns, combining finance data from Yahoo and query data from
Google: 

* `date` represents trading days between 2012 and 2017.

* `googLVL` represents an index on how much "Volkswagen" was googled.

* `volLVL` represents the total Volume traded for that specific day.

* `dif_highlowLVL` represents the difference between the day's highest and lowest stock price. You can think of it as proxy for volatility.

* `googRET` represents the log returns of `googLVLs`.

* `daxRET` represents the log returns of the DAX. It is a proxy for the market movements.

* `y_closeRET` represents our target variable as a log return. It is the closing price for the Volkswagen AG stock for the specified date. We are using the closing and not the adjusted closing price because it is reasonable to assume that dividends and market split information are also represented in the Google signal (As a reminder: Adjusted closing prices are corrected for financial events, i.e. stock splits or dividend payments). 

We are using everything except `date`. We could also try to extract further features like
dummies or a seasonal effect. We save that for next time. Our data should incorporate some 
seasonal effects already. Nevertheless, be creative! 

{{< highlight python "style=emacs" >}}
# We read in the dataset 
data = read_csv("final_df_VW.csv")
data = data.iloc[:,1] # delete column you dont want to use for training here!
                      # We are deleteting date here.
{{< /highlight >}}

### Data Preprocessing
We are left with our six variables including our target. Before we start with our
model training we need two more steps. First, we include a `MinMaxScaler` from the `sklearn` package. It always makes sense to think about proper preprocessing such as normalization.
If we included the dataset without it, `volLVL` could completely dominate our traning and skew our model. It often makes sense if your features are on a similar scale. Please see below the function for the normalization. Notice that we also save the scaler, we are using! 
You can use it for example to reverse the scaling after your predictions. 
The second step is a little more involved but it is crucial for working with sequences in <a href="https://keras.io/layers/recurrent/">Keras</a>.

{{< highlight python "style=emacs" >}}
# function to normalize
def normalize(df):
    """ 
        Uses minMax scaler on data to normalize. Important especially for Volume and google_lvl
        @param df: data frame with all features
    """
    df = DataFrame(df)
    df.dropna(inplace = True)
    df = df.values
    df = df.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    norm = scaler.fit_transform(df)
    return scaler, norm
{{< /highlight >}}


## Keras
Before we get into the exciting part, a small introduction...

> "Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research." (<a href="https://keras.io/">Keras Documentation</a>)

Our examples use Keras with Python 3.5, running TensorFlow as a backend. We will build a 
small LSTM architecture to teach you the in and outs of Keras' sequence modelling. 
At this point, we expect you to understand what a recurrsive layer does and how it is
special. We will talk about hidden states and weight updates, truncated backpropagation, and other things. If all that is new to you, check out some of the very detailed blogs we recommended in the 'theory' part. 
There are many other blogs and examples which go rather slowly. Jason Brownlee's <a href="https://machinelearningmastery.com">blog</a> covers many ideas slowly for beginners.
This blog does try to provide a condensed application example, that does not only showcase the easiest one-to-one prediction. 
If you just came here accidentally without any idea of Keras the <a href="https://keras.io/#keras-the-python-deep-learning-library">30s guide</a> can bring you up to speed. For more involved file an issue to the Keras' <a href="https://github.com/keras-team/keras/issues">Github</a>. Most of the time they are happy to
help! 


### A little more preprocessing...
The most confusing thing for people starting to work with Keras' recurrent layers is getting used to the shape of the input matrix. In contrast to a standard MLP, recurrent networks's input has an additional dimension. 

* The input matrix is 3D, where the first dimension is the number of samples in you batch (normally denoted as  `batch_size`). You can think of it as the number 
of rows of you input data after which you want your weights to be updated (careful: weights are not states). A higher `batch_size` reduces your computational time by reducing the number of updates. In many cases, especially if you are short on training data, you would set this to 
one and just update your weights after every sample. We will do that for our stock prediction,
since we only end up with about 1500 training days. If you had a different use case (i.e., Natural Language Processing) it could be beneficial to update weights only every 5 samples (i.e., `batch_size` = 5). 

* The second dimension represents the new time domain (`timesteps`). `Timesteps` define the number of steps in the past you are unfolding your recurrent layer. They define where the backpropagation is truncated to. It is important to understand that the longer your sequence per sample (more `timesteps`) is the more computational expensive your optimization gets, since the gradient is computated for every defined `timestep`. When you are used to auto-regressiv statistical modelling techniques, `timesteps` are difficult to understand. In a standard feed forward neural network (FFNN) or i.e. ARIMA setup, it would be natural to include your `timesteps` (lags) as `features`. Here is were many people struggle. In the LSTM, the right way to handle time dependencies is in the second dimensions. For stock market prediction it is crucial to find well defined time dependencies. If we set this to i.e. seven, every `features` would backpropate one week, with 30 one month etc. Technically, it is also possible to include different `features` with different `timesteps`. Missing steps would be padded with 0.
Would that be a problem? Most likely not since the model should learn to ignore them. 

* The last dimension represent `features`. There are six in our stock price example if we 
want to include the target variable also as a `feature`. This is the same as in FFNN.
<br>

Now that you understand the `batch_input_shape` (`batch_size`, `timesteps`, `features`) of a recurrent layer, you might have noticed that our dataset hardly has the correct dimensions to be fed into the model. Below you can see the function that changes that. The idea is simple: We shift the input and append it to the old dataset. We extend the dataset according to our specified `timesteps`.

{{< highlight python "style=emacs" >}}
# Append with timesteps
def createTimeSteps(df, lags=1):
    """ 
        creates the amount of timesteps from the target and appends to df. 
        How many lags we use to predict the target.
        @param df: data frame with all features
        @param lags: number of lags from the target that are appended
    """
    df = DataFrame(df)
    columns = list()
    for i in range(lags, 0, -1):
        columns.append(df.shift(i))
    columns.append(df) #add original
    # combine
    output = pd.concat(columns, axis=1)
    # replace rows with NaN values
    output.fillna(0, inplace = True)
    return output
{{< /highlight >}}

Now we're good to go. We are using our loaded dataset from above, apply `normalize`,
extend by our `timesteps`, split into training and test set with `TRAINING_DAYS`,
and choose our `features` and our `y`. It is good practive to define 
CONSTANTS in capital letter in the beginning of your training. It helps you to keep eveything structured and is very convenient for testing different setups.

{{< highlight python "style=emacs" >}}
# Everything prepared...
scaler, normalized_data = normalize(data)

BATCH_SIZE = 1 # batch size during training
TS = 14 # length of Sequence we use for our samples (7 = week, 30 = month)
FEATURES = 6 # number of features in data set
TRAINING_DAYS = 1250 # Training/Test split for data

full_df = createTimeSteps(normalized_data, TS) 
full_df = full_df.values # Training vs Test

train = full_df[:TRAINING_DAYS, :]
test = full_df[TRAINING_DAYS:, :]

input_var = int(TS*FEATURES) # Every feature has as many columns as defined timestep 
target = -1 # Our Volkswagen AG stock price is the last column of our dataset
X_train, y_train = train[:, :input_var], train[:, target]
X_test, y_test = test[:, :input_var], test[:, target]

X_train = X_train.reshape(TRAINING_DAYS, TS, FEATURES) 
X_test = X_test.reshape(X_test.shape[0], TS, FEATURES)
{{< /highlight >}}

Similar to any Keras network we can design recurrent architectures.
Just add an LSTM layer instead of a normal dense layer. If you call the function,
make sure that your input dimensions fit our dataset. Otherwise you will not be 
able to train your model. The code itself should be self-explanatory. 
Our first model is very easy. If you do not know anything about Keras,
please refer to the <a https://keras.io/#keras-the-python-deep-learning-library">30s guide</a>.
This should just give us a starting point to explain different concepts and extentions.  

{{< highlight python "style=emacs" >}}
# Our first very easy model
def helloModel(timesteps, features, batch_size=1):
    model = Sequential()
    model.add(LSTM(16, input_shape=(timesteps, features)))
    model.add(Dense(1))
    model.add(Activation('linear'))  
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
    return model
{{< /highlight >}}

Our `helloModel` has only one layer with 16 hidden neurons. It passes its input to
the dense layer which produces a one-step-ahead forecast. The first extention we would like to
introduce is `return_sequence`:

* In Keras when `return_sequence` = False:
The input matrix of the first LSTM layer of dimension (`nb_samples`, `timesteps`, `features`) will produce an output of shape (`nb_samples`, 16),
and only output the result of the last `timesteps` training.

* In Keras when `return_sequence` = True:
Also the output shape would be 3D (`nb_samples`, `timesteps`, `features`) for such a layer, since a output is saved after every `timesteps`. This gives us to extend our model in two different ways. First, we can start stacking LSTM layers together, since every previous LSTM layer also produces a 3D output. Additionally, we can make the model predict many-to-many.
If we specify `return_sequence` = True for the last layer it will produce 3D predictions (Careful: If you would like to apply another layer to every `timesteps` and not only to the last one, you need to use a <a href="https://keras.io/layers/wrappers/">TimeDistributed wrapper</a>).

{{< highlight python "style=emacs" >}}
# Our return model
def returnModel(timesteps, features, batch_size=1, return_sequence = False):
    model = Sequential()
    model.add(LSTM(32, input_shape=(timesteps, features), return_sequence = True ))
    model.add(LSTM(16, input_shape=(timesteps, features), return_sequence = True ))
    model.add(LSTM(8, input_shape=(timesteps, features), return_sequence = return_sequence ))
    if return_sequence:
        model.add(Dense(1))
    else:
        model.add(TimeDistributed(Dense(1)))
    model.add(Activation('linear'))  
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
    return model
{{< /highlight >}}

We are stacking three different LSTM layers and included the option to predict 
many-to-many, applying a Dense last layer to every `timesteps`. We have to be careful
here since also our target `y` should be a matrix now. If we extend `y` by the same `timesteps` as our input matrix, you can think of the prediction as a many-to-many lagged by one each.




<script>
setTimeout(function(){
    //We had an issue with the python comments having an extra newline character.
    $("code.language-python span").filter(function () { 
        return this.style.fontStyle == 'italic' && this.style.color == "rgb(0, 136, 0)";
    }).each(function(i,e){        
        e.innerText = e.innerText.replace("\n","");
    });
}, 0);
</script>

<style>
div.highlight {
margin: 25px 0 25px 0;
}
</style>