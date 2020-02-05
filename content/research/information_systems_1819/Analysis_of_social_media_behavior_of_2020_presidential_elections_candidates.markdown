---
layout: "post"
title: "Blogpost"
date: "2020-01-25 18:25"
---
+++
title = "Analysis of social media behavior of the 2020 presidential election candidates"
date = '2020-02-07'
tags = [ "Fasttext", "CNN", "Class19/20", "Sentiment Analysis"]
categories = ["course projects"]
banner = "img/seminar/sample/hu-logo.jpg"
author = "Seminar Information Systems (WS19/20)"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = "This blog post analyzes the tweets of the 2020 presidential candidates using Fasttext and CNN"
+++

#### Authors: Georg Velev, Iliyana Pekova

## The Research problem


```Python
plot_barplot(data_2019,"Descriptive Statistics: 2019")
```
<img style=" width:500px;display:block;margin:0 auto;"
src="/blog/static/img/seminar/NLP_analysis/Capture.PNG">

The aim of the current project is to firstly examine the way presidential candidates talk about topics of high social importance. Afterwards the public opinion in regard of the same topics is considered. Finally, a presidential campaign manual based on the results of the performed analysis is created. The social media platform used for the purpose is Twitter.
The presidential candidates whose tweets the project is based on, are the following ones:
1. Republican Party: Donald Trump, Joe Walsh, Bill Weld
2. Democratic Party: Cory Booker, Elizabeth Warren, Joe Biden, Bernie Sanders
The explored topics of interest are gun control, climate change, legalization of marijuana and immigration.

## Twitter Data Retrieval

### GetOldTweets3 vs Tweepy

The following section will briefly present the not so popular data retrieval package GetOldTweets3. The most widely used library for this purpose is Tweepy, which is developed by Twitter. However, GetOldTweets3 suits better the needs of the current project. The project requires the tweets of the presidential candidates from the past four years. Tweepy only allows the extraction of tweets from the past seven days. GetOldTweets3 on the other hand has no limitation regarding the amount of the retrieved data. In addition, GetOldTweets3 requires no authentification data like API-key and tockens (unlike Tweepy) and is robust and fast, making "try-except" blocks unnecessary (unlike in the case of Tweepy). Within this context is must be mentioned that Tweepy has various other advantages towards GetOldTweets3, but these are simply not relevant for this project. Some of them are the Tweepy's wide functionality like retrieving the most extensive information provided for each tweet or posting tweets, sending and receiving messages and following users through the API. When it comes to amount of accessible data GetOldTweets3 is inarguably the better library.

### Descriptive Statistics of retrieved Tweets

## Methodology

### Retrieval of pre-labeled Twitter Data

The starting point of the project is the retrieval of three (already) pre-labeled datasets, which serve as a train and test base of the later implemented machine learning algorithms.

#### Noisy pre-labeled Twitter Data

The first pre-labeled dataset contains noisy pre-labeled Twitter data. The definition as "noisy" refers to the fact that the data has been labeled solely according to the emojis contained in the tweets. The possible labels for this dataset are positive, negative and neutral. The initial training and test set have size of 160 000 and 498 tweets respectively. This ratio is indeed unreasonable, but this is not of any importance in this case (explanation follows in the section "Combined pre-labeled Twitter Data" below). The occurences of the labels look as follows: 800 182 positive, 800 177 negative and 139 neutral, which makes the data imbalanced. The neutral labels are so few that they are practically neglectable. Thus, they are removed and the dataset is left with its 1 600 359 remaining positive or negative labels.

#### Pre-labeled Twitter Data from the Python Library nltk

The next pre-labeled dataset is natively integrated in the python library nltk and thus retrieved directly from it. The dataset contains 5000 twitter samples labeled as negative and 5000 labeled as positive (again any separation in a train and test set does not currently matter).  

#### Pre-labeled Twitter Data from Kaeggle

The last pre-labeled dataset containing labeled tweets is the training set from a Kaeggle sentiment classification challenge. The training data contains 7086 sentences, already labeled as a positive or negative sentiment. Since it is a challenge a labeled test set is not provided, but (as already mentioned multiples times above) also not needed for the current project. The purpose of this dataset is to provide additional pre-labeled Twitter data.

#### Combined pre-labeled Twitter Data

Finally, all of the three datasets introduced above are combined into a new (sofar non-existing) labeled dataset. This is done because the machine learning algorithms do not deliver plausible results when trained on any of the original datasets described above. The new data consists of 10000 (5000 positive and 5000 negative) twitter samples from the nltk dataset, 10000 (5000 positive and 5000 negative) twitter samples from the noisy pre-labeled dataset and 5950 (2975 positive and 2975 negative) twitter samples from the Kaeggle challenge train set. The new dataset is perfectly balanced and has 12975 negative and 12975 positive labels. After being composed, the new pre-labeled dataset is split into a train and test dataset with a ratio of 70% - 30%. All machine learning training and testing is then performed on this data.

### Machine Learning Algorithms used for the Text Classification

The first algorithm this project implements is a convolutional neural network (CNN). The most often application of CNNs is in the field of image recognition, where an image is decomposed in its pixels, which together form a matrix. A kernel (filter, feature detector) then convolves over the matrix, which results into a feature map of the pixel matrix. The process is visualized in Figure 1.

![Figure 1](http://deeplearning.stanford.edu/wiki/images/6/6c/Convolution_schematic.gif)

**Figure 1**

In the case of natural language processing the process is similar in its nature. Instead of having a pixel matrix resulting from an image, one has an embedding matrix resulting from a natural language sentence. Similarly to the kernel convolving over the pixel matrix, the kernel in the context of NLP convolves over the embedding matrix. An important difference is that as shown in Figure 1, the kernel slides along both dimensions (x and y) of the matrix. Thus, the kernel itself is also two-dimensional, meaning its size is defined by two numbers. This is due to the fact the location of the single pixels within the pixel matrix is often of high relevance. Also similarities/differences neighboring pixels also reflect in similarities/difference in the actual image, so it makes sense to let the kernel convolve over multiple neighboring pixels in both dimensions. Figrue 2 is a step-by-step visualization of the convolutional process when having an emdding matrix as input. Each row of the embedding matrix is a numerical representation of a word from the input text. Unlike in the case of image recognition, a similarity/difference between neighboring numbers, contained in the same row of the embedding matrix, is of low relevance, since one cannot base any assumptions on it. Accordingly,

![Figure 2](http://www.wildml.com/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-12.05.40-PM-1024x937.png)

**Figure 2**
