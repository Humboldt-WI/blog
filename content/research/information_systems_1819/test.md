---
title: "Test"
date: 2019-02-07T17:06:29+01:00
draft: true
---

+++
title = "Text Classification with Hierarchical Attention Network"
date = '2019-02-08'
tags = [ "Deep Learning", "Neural Networks", "Class18/19", "Hierarchical Network", "NLP", "Classification", "Attention"]
categories = ["course projects"]
banner = "img/seminar/HAN_img/banner.jpeg"
author = "Seminar Information Systems (WS18/19)"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = "Hierarchical Attention Networks - An Introduction"
+++

# Text Classification with Hierarchical Attention Networks
## How to assign documents to classes or topics

#### Authors: Maria Kränkel, Hee-Eun Lee - Seminar Information System 18/19
_____________________________________________________________________

After reading this blog post, you will know:

* What text classification is and what it is used for
* What hierarchical attention networks are and how their architecture looks like
* How to classify documents by implementing a hierarchical attention network

## Introduction

**Imagine you work for a company** that sells cameras and you would like to find out what customers think about the latest release. Ratings might not be enough since users tend to rate products differently. Nowadays, you will be able to find a vast amount of reviews on your product or general opinion sharing from users on various platforms, such as facebook, twitter, instagram, or blog posts.

One might consider a product they rate with 3 out of 5 stars very good, others always give full stars even if they dislike a few aspects. Text classification can give a clue, whether ratings actually describe the overall opinion towards the product.  As you can see, the number of platforms that need to be operated is quite big and therefore also the number of comments or reviews. So, how can you deal with all of this textual data and gain important insights from it?

<br>
<br>
<img align="center" width="700" height="450"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/intro1.png">

## Outline
* [Introduction](#introduction)
* [Text Classification](#text-classification)
* [Applications](#applications)
* [Literature Review](#literature-review)
* [Text Classification with Hierarchical Attention Networks](#text-classification-with-hierarchical-attention-networks)
  * [Architecture of Hierarchical Attention Networks](#architecture-of-hierarchical-attention-networks)
  * [Word Level](#word-level)
  * [Sentence Level](#sentence-level)
  * [Implementation](#implementation)
* [News Classification](#news-classification)
* [Take Away](#take-away)

## Text Classification

**Evaluating all of the textual data manually** is very time consuming and strenuous. A more efficient way to extract important information from it is text classification. <br>
Text classification is a fundamental task in natural language processing. The goal is to assign unstructured documents (e.g. reviews, emails, posts, website contents etc.) to one or multiple classes. Such classes can be review scores, like star ratings, spam vs. non-spam classification, or topic labeling. <br>
Essentially, text classification can be used whenever there are certain tags to map to a large amount of textual data. To learn how to classify, we need to build classifiers which are obtained from labeled data. In this way, the process of examining information becomes automated and thus simpler.
<br>
<br>

## Applications

Text classification finds a variety of application possibilities due to large amount of data which can be interpreted.
<br>

By **topic labeling** every kind of assigning text to topics or categories is meant. This can also include unstructured texts. The main goal is to extract generic tags. Topic labeling is the most important and widest used application of text classification. It has a few sub-applications:

* **Marketing**: The 'new' marketing has moved from search engines to social media platforms where real communication between brands and users take place. Users do not only review products but also discuss them with other users. With text classification, businesses can monitor and classify users based on their online opinions about a product or brand. Based on this, trends and customer types (e.g. promoters or detractors) can be identified.
* **Reviews**: With text classification businesses can easily find aspects on which customers disagree with their services or products. They do not have to go through low rating reviews by themselves but can detect categories in which their product did or did not satisfy.  
* **Tagging content**: Platforms, like blogs, live from publications of many people or pool products from other websites. So, if these are not tagged thoroughly in the first place, there might be the need to tag these texts or products in order to simplify navigation through the website. User experience is improved by this application too. In addition, good classified and tagged websites are more likely to appear in search engines like Google. <br>
Mentioning Google: If you're using Gmail, your emails are already automatically filtered and labeled by Google's text classification algorithms. <br>

Another application is **sentiment analysis**. Imagine again how different customers might rate a product. Someone could be disappointed about one single feature and consequently give it a low star rating although they like the overall product. Or ratings might be low due to bad customer service whilst the product itself is satisfying. Text classification helps to identify those criteria.
Sentiment analysis predicts the sentiment towards a specific characteristic on the base of text classification. This not only finds economic application, but also for social and political debates. <br>

Text classification is already used for simpler applications, such as **filtering spam**. Also, a team of Google invented a method called Smart Replies in 2016. This method takes emails as inputs, identifies the sentiment or topic of the mailed text and automatically generates short, complete responses.  
<br>
<br>

## Literature Review

**For our implementation of text classification**, we have applied a hierarchical attention network, a classification method from Yang et al. from 2016. The reason they developed it, although there are already well working neural networks for text classification, is because they wanted to pay attention to certain characteristics of document structures which have not been considered previously. <br>
But before going deeper into this, let's have a look at what others did:

The basis of all text classification problems lies in the so-called Information Retrieval (IR) methods which were first developed in the early 1970s. These first methods were unsupervised which means that they try to find information from a given text document without classifying it or assigning labels to it in any kind. <br>

**Basic algorithms for IR** are:

  * Bag of Words (BoW): represents texts by frequency of appearing words
  * Term Frequency / Inverse Document Frequency (TF-IDF): sets term frequency and inverse document frequency in ratio and in this way represents texts by relevance of appearing words
  * N-grams: a set of co-occurring words (e.g. names)

Here you can see the most important steps in unsupervised text classification:

**Unsupervised**

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;width: 100%;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l"><strong>Year</strong></th>
    <th class="tg-yw4l"><strong>Authors</strong></th>
    <th class="tg-yw4l"><strong>Method Characteristics</strong></th>
	<th class="tg-yw4l"><strong>Paper</strong></th>
  </tr>
  <tr>
    <td class="tg-yw4l">1971</td>
    <td class="tg-yw4l">Jardine, van Rijsbergen</td>
    <td class="tg-yw4l">Clustering keywords of similar texts</td>
	<td class="tg-yw4l">[The Use of Hierarchic Clustering in Information Retrieval](https://www.researchgate.net/publication/220229653_The_Use_of_Hierarchic_Clustering_in_Information_Retrieval)</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;" class="tg-yw4l">1974</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Salton et al.</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Ranking words in accordance with how well they are able to discriminate the documents</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">[A Theory of Term Importance in Automatic Text Analysis.](https://eric.ed.gov/?id=ED096987)</td>

  </tr>
  <tr>
    <td class="tg-yw4l">1983</td>
    <td class="tg-yw4l">Salton, McGill</td>
    <td class="tg-yw4l">SMART - First text representations with vectors</td>
	<td class="tg-yw4l">[Introduction to Modern Information Retrieval](http://www.information-retrieval.de/irb/ir.part_1.chapter_3.section_6.topic_6.html)</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">1992</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Cutting et al.</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">SG - Scattering text to few clusters, manual gathering to sub-collections</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">[Scatter/Gather: A Cluster-based Approach to Browsing Large Document Collections](https://www-users.cs.umn.edu/~hanxx023/dmclass/scatter.pdf)</td>
  </tr>
  <tr>
    <td class="tg-yw4l">1998</td>
    <td class="tg-yw4l">Zamir, Etzioni</td>
    <td class="tg-yw4l">Suffix Tree - Phrases shared between documents - First IR for web search engines</td>
	<td class="tg-yw4l">[Web Document Clustering: A Feasibility Demonstration](https://homes.cs.washington.edu/~etzioni/papers/sigir98.pdf)</td>
  </tr>
</table>
<br>
<br>
Find all listed abbreviations in the following table:

<img align="center" width="350" height="100"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/Abb1.png">

<br>
With the improvement of the user-friendliness and related spread of internet usage, automated classification of growing numbers of data became important. Several supervised respectively semi-supervised methods (where the class information are learned from labeled data) are shown in the next table. <br>

<br>
**(Semi-) Supervised**
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;width: 100%;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l"><strong>Year</strong></th>
	<th class="tg-yw4l"><strong>Method</strong></th>
    <th class="tg-yw4l"><strong>Authors</strong></th>
    <th class="tg-yw4l"><strong>Characteristics</strong></th>
	<th class="tg-yw4l"><strong>Paper</strong></th>
  </tr>
  <tr>
    <td class="tg-yw4l">1995</td>
	<td class="tg-yw4l">PN</td>
    <td class="tg-yw4l">Makoto, Tokunaga</td>
    <td class="tg-yw4l">Clustering by maximum Bayesian posterior probability</td>
	<td class="tg-yw4l">[Hierarchical Bayesian Clustering for Automatic Text Classification](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=BE17EC88A0C9DB576DA0E36C70F54EC3?doi=10.1.1.52.2417&rep=rep1&type=pdf)</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;" class="tg-yw4l">1999</td>
	<td style="background-color: #f7f7f7;" class="tg-yw4l">PN</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Nigam et al.</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Improving learned classifiers with Expectation Maximization and Naive Bayes</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">[Text Classification from Labeled and Unlabeled Documents using EM.](http://www.kamalnigam.com/papers/emcat-mlj99.pdf)</td>
  </tr>
  <tr>
    <td class="tg-yw4l"> </td>
	<td class="tg-yw4l"> </td>
    <td class="tg-yw4l"> </td>
    <td class="tg-yw4l"> </td>
	<td class="tg-yw4l"> </td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">1998</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">SVM</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">Joachims</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Binary classifying, representation with support-vectors</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">[Text Categorization with Support Vector Machines: Learning with Many Relevant Features.](http://web.cs.iastate.edu/~jtian/cs573/Papers/Joachims-ECML-98.pdf)</td>
  </tr>
  <tr>
    <td class="tg-yw4l">2004</td>
	<td class="tg-yw4l">SVM</td>
    <td class="tg-yw4l">Mullen, Collier</td>
    <td class="tg-yw4l">Classifying with SVM and unigrams</td>
	<td class="tg-yw4l">[Sentiment analysis using support vector machines with diverse information sources](http://www.aclweb.org/anthology/W04-3253)</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">2005</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">SVM-H</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Matsumoto et al.</td>	
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">SVM + unigrams and bigrams, Sentence dependancy sub-trees, word sub-sequences</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">[Sentiment Classification Using Word Sub-sequences and Dependancy Sub-trees](https://link.springer.com/content/pdf/10.1007%2F11430919_37.pdf)</td>
  </tr>
  <tr>
    <td class="tg-yw4l"> </td>
	<td class="tg-yw4l"> </td>
    <td class="tg-yw4l"> </td>
    <td class="tg-yw4l"> </td>
	<td class="tg-yw4l"> </td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;" class="tg-yw4l">1994</td>
	<td style="background-color: #f7f7f7;" class="tg-yw4l">NN</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Farkas</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">NN + Thesaurus -> First weighted, dictionary-based relations</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">[Generating Document Clusters Using Thesauri and Neural Networks](https://vdocuments.site/ieee-proceedings-of-canadian-conference-on-electrical-and-computer-engineering-58e24c154d826.html)</td>
  </tr>
  <tr>
    <td class="tg-yw4l">1996</td>
	<td class="tg-yw4l">NN-SOM</td>
    <td class="tg-yw4l">Hyötyniemi</td>
    <td class="tg-yw4l">Competitive learning instead of error-correction (e.g. backpropagation), Mapping to reduced dimensions</td>
	<td class="tg-yw4l">[Text Document Classification with Self-Organizing Maps](http://lipas.uwasa.fi/stes/step96/step96/hyotyniemi3/)</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">1998</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">NN-SOM-H</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">Merkl</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">SOM on hierarchical document levels</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">[Text classification with self-organizing maps: Some lessons learned](https://www.sciencedirect.com/science/article/pii/S0925231298000320)</td>
  </tr>
  <tr>
    <td class="tg-yw4l"> </td>
	<td class="tg-yw4l"> </td>
    <td class="tg-yw4l"> </td>
	<td class="tg-yw4l"> </td>
	<td class="tg-yw4l"> </td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">2014</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">CNN</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Johnson, Zhang</td>	
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">CNN on word order instead of low-dimensional word vectors</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">[Effective Use of Word Order for Text Categorization with Convolutional Neural Networks](https://arxiv.org/pdf/1412.1058.pdf)</td>
  </tr>
<tr>
    <td class="tg-yw4l">2014</td>
	<td class="tg-yw4l">CNN</td>
    <td class="tg-yw4l">Kim</td>
    <td class="tg-yw4l">Simple CNN with static vectors on top of pre-trained word vectors</td>
	<td class="tg-yw4l">[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;" class="tg-yw4l">2014</td>
	<td style="background-color: #f7f7f7;" class="tg-yw4l">CNN-char</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">dos Santos, Zadrozny</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Semantic word information with CNN on character level</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">[Learning Character-level Representations for Part-of-Speech Tagging](http://proceedings.mlr.press/v32/santos14.pdf)</td>
  </tr>
  <tr>
    <td class="tg-yw4l">2015</td>
	<td class="tg-yw4l">LSTM</td>
    <td class="tg-yw4l">Tai et al.</td>
    <td class="tg-yw4l">LSTM on tree-structured networks</td>
	<td class="tg-yw4l">[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://aclweb.org/anthology/P15-1150)</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">2015</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">C-LSTM</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">Zhou et al.</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">CNN for higher-level representations, LSTM for sentence representations</td>
	<td style="background-color: #f7f7f7;"  class="tg-yw4l">[A C-LSTM Neural Network for Text Classification](https://arxiv.org/pdf/1511.08630.pdf)</td>
  </tr>
  <tr>
    <td class="tg-yw4l">2015</td>
	<td class="tg-yw4l">CNN/LSTM-GRU</td>
    <td class="tg-yw4l">Tang et al.</td>
    <td class="tg-yw4l">CNN / LSTM for sentence representation, GRU for semantic information</td>
	<td class="tg-yw4l">[Document Modeling with Gated Recurrent Neural Network for Sentiment Classification](http://ir.hit.edu.cn/~dytang/paper/emnlp2015/emnlp2015.pdf)</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">2016</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">HAN</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">Yang et al.</td>	
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">GRU and attention on contexts on hierarchical document level</td>
    <td style="background-color: #f7f7f7;"  class="tg-yw4l">[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)</td>
  </tr>
</table>
<br>

<img align="center" width="300" height="340"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/Abb2.png">
<br>
<br>
Nigam et al.(1999) show in their probabilistic method that text classification improves significantly when learning from labeled data. <br>
Support Vector Machines (SVM) use support vectors as classifier. Here, Matsumoto et al.(2005) involve hierarchical structure by creating sentence representations. <br>
Since we use a neural network, the comparison with other neural networks is a priority to us. Of course, there are several different implementations of convolutional and recurrent neural networks;  selected steps during the development of NN for text classification are mentioned in the table. <br>
It is already common use to combine layers of CNN and RNN. Several approaches successfully covered the hierarchical structure of documents (e.g. Zhou et al., 2015) and computed importance weights. Still, contexts of words and sentences, including the changing meanings in different documents, are new to text classification tasks and find a first solution with HAN. 
<br>
<br>

## Text Classification with Hierarchical Attention Networks

Contrary to most text classification implementations, a Hierarchical Attention Network (HAN) also considers the hierarchical structure of documents (document - sentences - words) and includes an attention mechanism that is able to find the most important words and sentences in a document while taking the context into consideration. Other methods only return importance weights resulting from previous words. <br>
Summarizing, HAN tries to find a solution for these problems that previous works did not consider:

* Not every word in a sentence and every sentence in a document are equally important to understand the main message of a document.

* The changing meaning of a word depending on the context needs to be taken into consideration. For example, the meaning of the word "pretty" can change depending on the way it is used: "The bouquet of flowers is pretty" vs. "The food is pretty bad".

In this way, HAN performs better in predicting the class of a given document. <br>
To start from scratch, have a look at this example:
<br>
<br>
<img align="center" width="350" height="100"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/reviewyelp.png">
<br>
Here, we have a review from yelp that consists of five sentences. The highlighted sentences in red deliver stronger meaning compared to the others, and inside, the words *delicious* and *amazing* contribute the most in attributing the positive attitude contained in this review. This example reproduces our aforementioned statement about HAN, which we also intuitively know: not all parts of a document are equally relevant to gain the essential meaning from it.
<br>
<br>

### Architecture of Hierarchical Attention Networks

**This is how the architecture** of HANs looks like:
<img align="center" width="430" height="450"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/han_architecture.jpg">

<br>
First, the network considers the hierarchical structure of documents by constructing a document representation by building representations of sentences and then aggregating those into a document representation. <br>
**Sentence representations** are built by first encoding the word of a sentence and then applying the attention mechanism on them resulting in a *sentence vector*. <br>
**Document representation** is built in the same way, however, it only receives the sentence vector of each sentence of the document as input. <br>

To understand the different processes of the HAN architecture better, we took the structure a little bit apart and provide a different perspective. Take a look: <br>
The same algorithms are applied twice: First on word level and afterwards on sentence level. <br>
The model consists of

* the encoder, which returns relevant contexts, and

* the attention mechanism, which computes importance weights of these contexts as one vector.
<br>
<br>

#### Word Level
<img align="center" width="750" height="320"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/han_word.png">
<br>
<br>
<img align="center" width="180" height="25"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/xit.png">
	 
* As input we have the structured tokens **$w_{it}$** which represent the word i per sentence t. We do not keep and process all words in a sentence. Learn more about that in the section [data preprocessing](data-preprocessing).
* Since the model is not able to process plain text of the data type *string*, the tokens run through an Embedding layer which 'assigns' multidimensional vectors **$W_{e}$** **$w_{it}$** to each token. In this way, words are represented numerically as **$x_{it}$** as a projection of the word in a continuous vector space. <br>
* There are several embedding algorithms: the most popular ones are [word2vec](https://code.google.com/archive/p/word2vec/) and [GloVe](https://nlp.stanford.edu/projects/glove/). It is also possible to use pre-trained word embeddings, so you can accelerate your model training. <br>
<br>

##### Word Encoder

* These vectorized tokens are the inputs for the next layer. Yang et al.(2016) use a Gated Recurrent Network (GRU) as an encoding mechanism. 
* As a short reminder: In an RNN, states are 'remembered' to ensure we can predict words depending on previous words. A GRU has a so-called 'hidden state' which can be understood as a memory cell to transfer information. Two gates decide whether to keep or forget information and with this knowledge, to update the information that should be kept. If you are interested in learning more about GRU, have a look at this nice [blog post](https://isaacchanghau.github.io/post/lstm-gru-formula/). <br>
* The purpose of this layer is to extract relevant contexts of every sentence. We call these contexts *annotations* per word. <br>
 Note that in this model, a *bidirectional* GRU is applied to get annotations of words by summarizing information from both directions resulting in a summarized variable **$h_{it}$**.  <br>
<img align="center" width="180" height="110"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/wordEncoder.png">

##### Word Attention

* The annotations **$h_{it}$** build the base for the attention mechanism which starts with another hidden layer, a one-layer Multilayer Perceptron. The goal is to let the model learn through training with randomly initialized weights and biases. Those 'improved' annotations are then represented by **$u_{it}$**. Furthermore, this layer ensures that the network does not falter with a tanh function. This function 'corrects' input values to be between -1 and 1 and also maps zeros to near-zero. <br>

<img align="center" width="180" height="25"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/u_it.png">
	 
* Our new annotations are again multiplied with an outside trainable context vector **$u_{w}$** and normalized to an importance weight per word **$\alpha_{it}$** by a softmax function. <br>
 <img align="center" width="180" height="70"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/a_it.png">
	 
* The sum of these importance weights concatenated with the previously calculated context annotations is called sentence vector **$s_{i}$** <br>
<img align="center" width="130" height="50"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/s_i.png">
<br>

#### Sentence Level

<br>
<img align="center" width="700" height="350"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/han_sent.png">

<br>

* Then the whole network is run on sentence level with basically the same procedure as on word level, but now we focus on the sentence i. Of course, there is no embedding layer as we already get sentence vectors **$s_{i}$** from word level as input.
<br>

##### Sentence Encoder

* Contexts of sentences are summarized with a bidirectional GRU by going through the document forwards and backwards.
<br>
<br>
<img align="center" width="180" height="70"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/h_i.png">
<img align="center" width="130" height="35"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/h_ii.png">

##### Sentence Attention

* Trainable weights and biases are again outside randomly initialized.
* The final output is a document vector **$v$** which can be used as features for document classification.
<br>

<img align="center" width="180" height="150"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/ualphav.png">
<br>
<br>

### Implementation

We use the 'high-level neural networks API' [Keras](https://keras.io/) which is extremely useful for deep learning problems. We recommend to install it in an own python environment. <br>

#### Data Preprocessing

To demonstrate the application of HANs, we use Amazon reviews for electronic data which are publicly available [here](http://jmcauley.ucsd.edu/data/amazon/). This data set consists of nearly 1.7 million reviews. As the model learns through training, it is highly important to have data sets with a large number of observations. Nevertheless, a million reviews would take us **days** to train on, so we set the number of reviews to keep equal to 100,000.
<br>

**Words have to be lemmatized** to ensure that not every single typo or related term is handled by itself. Additionally, so-called stop words are filtered out. In our case, they are mainly prepositions like *as* or *to* that do not contribute to the meaning of the text. Have a look at function **cleanString**.

<script src="https://gist.github.com/kraenkem/fbcde89789b2f8a0d9fb094cb40a42fe.js"></script>

**After that, we can tokenize** the given sentences. We set the maximum number of words to keep equal to 200,000.

<script src="https://gist.github.com/kraenkem/473a99e16b04052f28c132623673abe2.js"></script>

<br>
We keep 62,837 tokens. <br>
A cleaned review looks like this. First, we have the original review and underneath the review without stop words and with lemmatization:
<br>

<script src="https://gist.github.com/leeh1234/0d4ce05bef2111efcfaa45784973c366.js"></script>

**For the vectorization of our tokens** we use one of GloVe's pre-trained embedding dictionaries with 100 dimensions which means that one word will be represented by 100 values in a matrix. As mentioned [before](#word-level), this accelerates our training. We match our tokens with the pre-trained dictionary and filter out words that appear rarely (mostly due to spelling mistakes). As you can see, reviewers for our chosen products do not pay attention to correct spelling. Therefore, we can only remain 20,056 words to proceed. This will influence the performance of our model. But we will come to this later.

<script src="https://gist.github.com/kraenkem/24e7770761bb1924fc61ae01a9b29493.js"></script>

For a better comprehension of what those embeddings mean, have a closer a look at an example sentence and a single token. *Great* is described by 100 values in vector spaces computed by for instance nearest neighbors.

<script src="https://gist.github.com/leeh1234/4aebbe9f19d7be410e038c3656b8a0b4.js"></script>

In the last step of data preprocessing, we want to set a train, validation and test data set. For that, we define the function **split_df** which ensures that all sets are balanced, hence they have the same ratio for each class as the full data set. Without this predefined grouping by star rating, it could happen that the model only trains on the most occurring rating.

<script src="https://gist.github.com/kraenkem/1488dba443356fbeebcccc134f980daa.js"></script>

<br>
#### Attention Mechanism

Before we can concatenate the layers of the network in Keras, we need to build the attention mechanism. Keras has a class '[Writing your own Keras layer](https://keras.io/layers/writing-your-own-keras-layers/)'. Here you are given some useful functions to implement attention. 

<script src="https://gist.github.com/kraenkem/827f39d18c24e43c44b55c8971dce3f2.js"></script>

The class **AttentionLayer** is successively applied on word level and then on sentence level.

* **init** initializes variables from a uniform distribution. Also, we set *supports_masking = True* because the network needs fixed input lengths. Therefore, every input gets a mask with the length equal to the maximum input length initialized with 0. Then, the mask will be padded with "1" at each position where the input holds values. Both padded input sequence and appropriate mask run through the network; to avoid that the network will recognize every input with the same length because of padding, the mask is used to check whether the input is just padded or actually has maximum length. 

* **build** defines the weights. We set *len(input_shape) == 3* as we get a 3d tensor from the previous layers.

* **call** builds the attention mechanism itself. As you can see, we have *$h_{it}$*, the context annotations, as input and get the sum of importance weights, hence, the sentence vector *$s_{i}$*, as output. In between, the current variable is reduced by the last dimension and expanded again because masking needs a binary tensor.
<br>
<br>

#### Model

Congrats, you made it through a huge mass of theoretical input. Now, let's finally see how the model performs. Some last little hints:

* The layers have to be combined on word and sentence level.

* *TimeDistributed* applies to all word level layers on each sentence.

* We want to have an output dimensionality of GRU equal to 50 because running it forwards and backwards returns 100 dimensions - which is the dimensionality of our inputs.

* *Dropout* is a regularizer to prevent overfitting by turning off a number of neurons in every layer. 0.5 gets a high variance, but you can play around with this as well as with other parameters.

* *Dense* implements another layer for document classification. The document vector runs again with outside weights and biases through a softmax function.

<script src="https://gist.github.com/kraenkem/a84e0ab8c14d98498276b479255e128b.js"></script>

**We train** the model throughout a relatively small number of 7 epochs since our input data are already pre-trained and could overfit after too many epochs. Also, the batch size of 32 works with a large number of inputs due to our large data set. Note that you have to train reviews **$x$** against labels **$y$** (in this case the 5-star ratings).

<script src="https://gist.github.com/kraenkem/b0f0bfbb1efdeec7d808b69beb521d0e.js"></script>
<script src="https://gist.github.com/kraenkem/e29cde258cb852f791516ef34cdc5775.js"></script>
<script src="https://gist.github.com/kraenkem/f54e681b995cefce72b641a12c50a88c.js"></script>

<br>
<br>

**Model evaluation** is with 69 % quite high how a comparison with the results from Yang et al.(2016) as well as from others shows (see table below).

**Also, history plots** show that the training data set perform pretty well. Still, this is unfortunately not supported by the validation data set. This might be because of the small number of words we proceeded with after the embedding layer which filtered out almost 70 % of all tokens due to misspelling. An improvement could be created with an even smaller batch and epoch size, or with a better, less mistaken data set.

<img align="center" width="700" height="500"
     style="display:block;margin:0 auto;"  
	 src="/blog/img/seminar/HAN_img/doc_class_comp_neu.JPG">
Note: HN-AVE and HN-MAX refer to hierarchical networks with averaging and max-pooling methods. HN-ATT refers to hierarchical attention networks as described in this blog.
<br>
<br>

This is how one of our 5-star reviews looks like that our model has predicted. The categorization of the review as a 5-star rating works quite well here: <br>
<script src="https://gist.github.com/leeh1234/8b3cbe430843f6abae3d21ccc82f7a5e.js"></script>
<br>
<br>

## News Classification

To further demonstrate the attention mechanism, we also implemented the HAN on news articles to be able to classify them into categories, as well as to gain short summaries of articles by extracting the most important sentences using sentence attention weights. News articles are particularly interesting within this context because the hierarchical architecture can fully exploit the length of the document, compared to shorter documents, such as tweets.<br>
We used a publicly available [data set](http://mlg.ucd.ie/datasets/bbc.html) from the British Broadcasting Corporation (BBC) which contains 2225 news articles from the BBC news website from 2004-2005. The news articles are sorted after five different categories: business, entertainment, politics, sport, and tech. 

### Parameters
As news articles tend to be longer than product reviews, we first calculated the average number of words in each sentence and the average number of sentences in each document. <br>
<script src="https://gist.github.com/leeh1234/1cb7820700c999b69f875d393f70865d.js"></script>
<br>
Subsequently, we adjusted the parameters and increased the maximum number of sentences in one document and the maximum number of words in each sentence. <br>
<script src="https://gist.github.com/leeh1234/1c7888cc8ca298ca0a2e071df8c761dc.js"></script>
<br>

### HAN Model

The data preprocessing steps are the same as for the Amazon data set. Hence, we skip right to our HAN Model which looks as follows. Merely, the parameters have changed here. <br>
<script src="https://gist.github.com/leeh1234/5b75b55aa6e5328a7ab996bf392ae689.js"></script>

<br>
<br>
We get the following results for our training, validation, and test set:
<br>
<script src="https://gist.github.com/leeh1234/74526125e260c69460a1cfd79146ae8e.js"></script>
<br>
<script src="https://gist.github.com/leeh1234/0cbcce6a9cbdcaf3122951cd2d522b63.js"></script>

Compared to the Amazon data set, the BBC data set exhibits a much higher accuracy rate. This is probably due to several facts: 

* News articles are in general much longer than product reviews, and therefore the HAN can exploit this and gain more information. 
* Also, news articles have no grammar and spelling mistakes, while product reviews written by users just burst from them. Grammar and spelling mistakes lead to misinterpretation of words and thus loss of information. 
* Another aspect is that the categorization classes of the BBC data set are much easier to distinguish, whereas the star rating categorization of Amazon is very subjective and it is quite hard to draw a straight line between different categories.   

### Input new articles
To access newly released articles from BBC, we need to scrape the BBC website and save the title and text which is then cleaned, as described in the preprocessing, and subsequently converted to a sequence of numbers (see: [embeddings](#Implementation).)
<script src="https://gist.github.com/leeh1234/3ecc73e0f3c2c163c8e0dea73f33981e.js"></script>

### Sentence Attention Model
Now, we need to build a new model to be able to extract the attention weights for each sentence. This is to identify the five most important sentences within a news article to put them together and create a short summary. <br>
To extract the attention weights that lie within the hidden layer of the attention mechanism, we build a separate model instead of using the complete attention model of the HAN. The sentence attention model encompasses the processes from the input of the HAN to the output of the attention layer on sentence level which is where we basically cut it off. We constructed *sent_coeffs* so that we not only gain the attention output from our model but also the attention weights. Thus, we redefined the model in order to obtain the attention weights for each sentence which we use to create a summary of the news article. <br>

<script src="https://gist.github.com/leeh1234/98553337afe357407002c5a698ac8a46.js"></script>

### Word Attention Model
Additionally, we want to extract the usually hidden word attention weights as well for which we need to build another model. The words with the most attention serve as a good overview or framework for the article. 

<script src="https://gist.github.com/leeh1234/6b5d21606a399d01fc1d5aca0d076469.js"></script>
<br>

### Output

This is how the final output for the BBC article that we scraped from the website looks like.
Our HAN model has successfully predicted the category of the article as *entertainment*. Moreover, we were able to extract the attention weights with the sentence and word attention models. Hence, we gained the five most important words which actually provide a good overview of the article and could be used as additional tags. The summary of the article which we obtained through the sentence attention weights, also shows a quite well-working abstract. <br>  
As a comparison, you can look at the full text below.

<script src="https://gist.github.com/leeh1234/07ac108692fe2afd5096a899e6ac4c25.js"></script>
<br>
<br>


The information of the articles could then be saved in a new database. The words with the most attention could be used as new tags for the database and could facilitate search navigation if also implemented on websites. This might be useful for news agencies that have to deal with many articles a day and need to hold on to information for research purposes. 

<script src="https://gist.github.com/leeh1234/13b13ff87bbc6f30e6c6669d3f9b063e.js"></script>

<br>
<br>

## Take Away

<img align="center" width="600" height="350"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/eternity.png">
<br>	 
As you can see, the hierarchical attention network is a well performing instrument to create some pretty cool text classification solutions. We hope this blog post - regardless of its mass of information - gives you an understanding of how to use HAN. The most relevant points to remember are: 

* the hierarchical structure of documents (document - sentence - word),
* paying attention to contexts of sentences and words,
* by considering changing contexts, HAN performs better for classification problems.
