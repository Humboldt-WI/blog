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
banner = "img/seminar/han/banner.jpg"
author = "Seminar Information Systems (WS18/19)"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = "Hierarchical Attention Network - An Introduction"
+++

# Text Classification with Hierarchical Attention Network
## How to assign documents to classes or topics

#### Authors: Maria Kränkel, Hee-Eun Lee - Seminar Information System 18/19
_____________________________________________________________________

After reading this blog post you will know:
* What text classification is and what it is used for
* How to classify documents with a hierarchical attention neural network
* How to implement a hierarchical attention network

## Introduction

**Imagine you work for a company** that sells cameras and you would like to find out what customers think about the latest release. Ratings might not be enough since users tend to rate products differently. One might consider a product they rates with 3 out of 5 stars very good, others always give full stars even if they dislike a few aspects. Text classification can give a clue, whether ratings actually describe the overall opinion towards the product. Additionally, the number of possibilities to get opinions from is rising: Nowadays, you will be able to find a vast amount of reviews on your product or general opinion sharing from users on various platforms, such as facebook, twitter, instagram, or blogposts. As you can see, the number of platforms that need to be operated is quite big and therefore also the amount of comments or reviews. So, how can you deal with all of this textual data and gain knowledge from it?

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
* [Text Classification with Hierarchical Attention Network](#text-classification-with-hierarchical-attention-network)
  * [Architecture of Hierarchical Attention Network](#architecture-of-hierarchical-attention-network)
  * [Word Level](#word-level)
  * [Sentence Level](#sentence-level)
  * [Implementation](#implementation)
* [News Classification](#news-classification)
* [Take Away](#take-away)

## Text Classification

**Evaluating all of the textual data manually** is very time consuming and strenuous. A more efficient way to extract important information from it is text classification. <br>
Text classification is a fundamental task in natural language processing. The goal is to assign unstructured documents (e.g. reviews, emails, posts, website contents etc.) to classes, that is, to classify them. Such classes can be review scores like star ratings, spam or topic labeling. <br>
Essentially, text classification can be used whenever there are certain tags to map to a large amount of textual data. To learn the how to classify, we need to build classifiers which are obtained from labeled examples. In this way, the process of examining information becomes automated and thus simpler.
<br>
<br>

## Applications

Text classification finds a variety of application possibilities due to the large amount of data which can be interpreted.
<br>

By **topic labeling** every kind of assigning text to topics or categories is meant. This can also include unstructured texts. The main goal is to extract generic tags. Topic labeling is the most important and widest used application of text classification. It has a few sub applications.

* **Marketing**: The 'new' marketing has moved from search engines to social media platforms where real communication between brands and users take place. Users don not only review products but also discuss them with other users. With text classification, businesses can classify those products which have great consideration. Based on this, trends and customer types are examined.

* **Reviews**: With text classification businesses can easily find aspects on which customers disagree with their services or products. They do not have to go through low rating reviews by themselves but can detect categories in which their product did or did not satisfy.  
* **Tagging content**: Platforms like blogs live from publications of many people or pool products from other websites. So, if these are not tagged thoroughly in the first place, there might be the need to tag these texts or products in order to simplify navigation through the website. User experience is improved by this application too. In addition, good classified and tagged websites are more likely to appear in search engines like Google. <br>
Mentioning Google: If you're using gmail, your mails are already automated filtered and labeled by Google's text classification algorithms. <br>

Another application is **sentiment analysis**. Imagine again how differently customers might rate a product. Someone could be disappointed about one single feature that they rate it low although they liked the overall product. Or ratings might be low due to bad customer service whilst the product itself is satisfying. Text classification helps to identify those criteria.
Now, sentiment analysis predicts the sentiment towards a specific characteristic on base of text classification. This not only finds economic application but especially in sentiment analysis for social and political debates. <br>

Text classification is already used for simpler applications like **filtering spam**. Also, a team of Google invented a method called Smart Replies in 2016. This method takes emails as inputs, identifies the sentiment or topic of the mailed text and automatically generates short, complete responses.  
<br>
<br>

## Literature Review
### How do different methods perform in text classification problems?
**For our implementation of text classification**, we have applied a hierarchical attention network, a classification method from Yang and others from 2016. The reason they developed it, although there are already well working neural networks for text classification, is because they wanted to pay attention to certain characteristics of document structures which have not been considered previously. <br>
But before going deeper into this, let's have a look at what others did:

The basics of all text classification problems lie in so-called Information Retrieval (IR) methods which started to be developed in the early 1970s. These first methods were unsupervised, that is, they try to find information from a given text document without classifying it or assigning labels to it in any kind. <br>

**Basic algorithms for IR** are:

  * Bag of Words: represents texts by frequency of appearing words
  * Term Frequency / Inverse Document Frequency: sets term frequency and inverse document frequency and in this way represents texts by relevance of appearing words
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

<img width="320" height="120"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/abbr_IR.JPG">

<br>
With the improvement of the user-friendliness and related spread of internet usage, automated classification of growing numbers of data became important. Several supervised respectively semi-supervised methods (where the class information are learned from labeled data) are shown in the next table. <br>
Since we use a neural network, the comparison with other neural networks is prior for us. Of course, there are several different implementations of convolutional and recurrent neural networks, below are only mentioned the most 'innovative'.

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

<img width="320" height="340"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/abbr_sup.JPG">

<br>
<br>

## Text Classification with Hierarchical Attention Network

Contrary to the most text classification implementations, HAN also considers the hierarchical structure of documents (document - sentences - words) and includes an attention mechanism that's able to find the most important words and sentences given the whole contexts, whereas other methods only return importance weights resulting from previous words. <br>
Summarizing, HAN tries to find a solution to these problems, previous works did not consider:
* Not every word in a sentence and every sentence in a document is equally important to extract the main information from the document.
* To extract the main information in changing contexts, it is not enough to work with the single information of each word, but to get the context of all sentences and word interactions.

In this way, HAN performs better in predicting the class of a given document. <br>
To start from scratch, have a look at this example:
<br>
<img align="center" width="350" height="120"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/reviewyelp.png">

<br>
<br>
Here we have a review from yelp that consists of five sentences. The highlighted sentences in red deliver stronger meaning compared to the others and inside, the words *delicious* and *amazing* contribute the most in attributing the positive attitude contained in this review. HAN predicts pretty well the most relevant information as it assorts with what we would intuitively gain from this review. <br>
<br>

### Architecture of Hierarchical Attention Network

**This is how the architecture** of HAN looks like:
<img align="center" width="430" height="450"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/han_architecture.jpg">

<br>
First, the network considers the hierarchical structure of documents by constructing a document representation by building representations of sentences and then aggregating those into a document representation. <br>
**Sentence representations** are built by encoding the word of a sentence firstly and apply the attention mechanism on them secondly resulting in a *sentence vector*. <br>
**Document representation** is built in the same way, only having the sentence vector of each sentence of the document as input. <br>

Now, have a different view on the architecture of the model to understand how it works. <br>
The same algorithms are applied two times: First on word level and afterwards on sentence level. <br>
The model consists of
* the encoder, which returns relevant contexts, and
* the attention mechanism, which computes importance weights of these contexts as one vector.

#### Word Level
<img align="center" width="750" height="320"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/han_word.png">

* As input we have structured tokens **w_it**, that is word i per sentence t. We do not keep all words in a sentence. Learn more about that in section [data preprocessing](data-preprocessing).
* Since the model is not able to process plain text of data type *string*, the tokens run through an Embedding layer which 'assigns' multidimensional vectors **W_e*w_ij** to each token. In this way, words are represented numerically as **x_it** as a projection of the word in a continuous vector space. <br>
	There are several embedding algorithms; the most popular are [word2vec](https://code.google.com/archive/p/word2vec/) and [GloVe](https://nlp.stanford.edu/projects/glove/). It is also possible to use pre-trained word embedding, so you can accelerate your model training. <br>
	
<img align="center" width="180" height="25"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/xit.png">

##### Word Encoder

* These vectorized tokens are the inputs for the next layer. Yang et al. use a Gated Recurrent Network (GRU) as encoding mechanism. As a short reminder: In a RNN, states are 'remembered' to ensure we can predict words depending on previous words. GRU has a so-called 'hidden state' which can be understood as a memory cell to transfer information. Two gates decide about whether to keep or forget information and with this knowledge, to update the information that should be kept. If you are interested in learning more about GRU, have a look at this nice [blog post](https://isaacchanghau.github.io/post/lstm-gru-formula/). <br>
	The purpose of this layer is to extract relevant contexts of every sentence. We call these contexts *annotations* per word. <br>
	Note that in this model, *bidirectional* GRU is applied to get annotations of words by summarizing information from both directions resulting in a summarized variable **h_it**.  <br>
<img align="center" width="180" height="110"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/wordEncoder.png">

##### Word Attention

* Those annotations h_it build the base for the attention mechanism which starts with another hidden layer, a one-layer Multilayer Perceptron. Goal is to let the model learn through training with randomly initialized weights and biases. Those 'improved' annotations are then represented by **u_it**. Furthermore, this layer ensures that the network does not falter with a tanh function. This function 'corrects' input values to being between -1 and 1 and also maps zeros to near-zeros. <br>

<img align="center" width="180" height="25"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/u_it.png">
* Our new annotations are again multiplied with an outside trainable context vector **u_w** and normalized to an importance weight per word **alpha_it** by softmax function. <br>
 <img align="center" width="180" height="70"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/a_it.png">
* The sum of these importance weights concatenated with the previously calculated context annotations is called sentence vector **s_i** <br>
<img align="center" width="130" height="50"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/s_i.png">


#### Sentence Level

<br>
<img align="center" width="700" height="350"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/han_sent.png">


* Then the whole network is run on sentence level with basically the same procedure as on word level but now we focus on the actual sentence i. Of course, there is no embedding layer as we already get sentence vectors **s_i** from word level as input.
<br>

##### Sentence Encoder

* Contexts of sentences are summarized with a bidirectional GRU by going through the document forwards and backwards.
<img align="center" width="180" height="70"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/h_i.png">
<img align="center" width="130" height="35"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/h_ii.png">
	 
##### Sentence Attention

* Trainable weights and biases are again outside randomly initialized.
* The final output is a document vector **v** which can be used as features for document classification.

<img align="center" width="180" height="150"
     style="display:block;margin:0 auto;" 
	 src="/blog/img/seminar/HAN_img/ualphav.png">
<br>
<br>

### Implementation

As the package [Keras](https://keras.io/) is 'a high-level neural networks API' extremely useful for deep learning problems, we recommend to install it in an own python environment. <br>

#### Data Preprocessing

To demonstrate how to apply HAN we use a part of Amazon reviews for Electronic data which are public available [here](http://jmcauley.ucsd.edu/data/amazon/). This data set consists of nearly 1.7 million reviews. As the model learns through training it is highly important to have data sets with a large number of observations. Nevertheless, a million reviews would take us **days** to train on, so we set the number of reviews to keep equal to 100,000.
<br>

**Words have to be lemmatized** to ensure that not every single typo or related term is handled by itself. Additionally, so-called stop words are filtered out. In our case, that is mainly prepositions like *as* or *to* that do not contribute to the meaning of the text. Have a look at function **cleanString**.

<script src="https://gist.github.com/kraenkem/fbcde89789b2f8a0d9fb094cb40a42fe.js"></script>

**After that we can tokenize** the given sentences. We set the maximum number of words to keep equal to 200,000.

<script src="https://gist.github.com/kraenkem/473a99e16b04052f28c132623673abe2.js"></script>

<br>
We remain 62,837 tokens. A cleaned and tokenized review looks like this:
<br>

<script src="https://gist.github.com/leeh1234/0d4ce05bef2111efcfaa45784973c366.js"></script>

**For vectorization of our tokens** we use one of GloVe's pretrained embedding dictionaries with 100 dimensions, that is one word is represented by 100 values in a matrix. As mentioned [before](#word-level), this accelerates our training. We match our tokens with the pretrained dictionary and filter out words that appear rarely (mostly due to spelling mistakes). As you can see, reviewers for our chosen products do not pay attention to correct spelling. Unfortunately, we therefore can only remain 20,056 words to proceed. This will influence performance of our model. But we will come to this later.

<script src="https://gist.github.com/kraenkem/24e7770761bb1924fc61ae01a9b29493.js"></script>

For a better comprehension of what those embeddings mean, have a closer a look at an example sentence and a single token. *Great* is described by 100 values in vector spaces computed by for instance nearest neighbors.

<script src="https://gist.github.com/leeh1234/4aebbe9f19d7be410e038c3656b8a0b4.js"></script>

In a last step of data preprocessing, we want to set a train, validation and test data set. For that, we define a function **split_df** which ensures that all sets are balanced hence they have the same ratio for each class as the full data set. Without this predefined grouping by star rating. it could happen that the model only trains on the most occurring rating.

<script src="https://gist.github.com/kraenkem/1488dba443356fbeebcccc134f980daa.js"></script>

<br>
#### Attention Mechanism

Before we can concatenate the layers of the network in Keras, we need to build the attention mechanism. Keras has a class '[Writing your own Keras layer](https://keras.io/layers/writing-your-own-keras-layers/)'. Here you are given some useful functions to implement attention. 

<script src="https://gist.github.com/kraenkem/827f39d18c24e43c44b55c8971dce3f2.js"></script>

Class **AttentionLayer** is applied successively on first word and then sentence level.

* **init** initializes variables from a uniform distribution. Also, we set *supports_masking = True* because the network needs fixed input lengths. If some inputs are shorter than maximum input length a mask will be created initialized with 0. Then the mask will be 'filled up' with 1 to positions where the input has values in. This is further defined in the next functions.

* **build** defines the weights. We set *len(input_shape) == 3* as we get a 3d tensor from the previous layers.

* **call** builds the attention mechanism itself. As you can see, we have h_it, the context annotations, as input and get the sum of importance weights, hence sentence vector s_i, as output. In between, the current variable is reduced by the last dimension and expanded again because masking needs a binary tensor.
<br>

#### Model

Congrats, you made it through a huge mass of theoretical input. Now, let's finally see how the model performs. Some last little hints:

* The layers have to be combined on word and sentence level.

* *TimeDistributed* applies all word level layers on each sentence.

* We want to have an output dimensionality of GRU equal to 50, because running it forwards and backwards returns 100 dimensions - which is the dimensionality of our inputs.

* *Dropout* is a regularizer to prevent overfitting by turning off a number of neurons in every layer - 0.5 gets a high variance, but you can play around with this as well as with other parameters.

* *Dense* implements a last layer for actual document classification. The document vector runs again with outside weights and biases through a softmax function.

<script src="https://gist.github.com/kraenkem/a84e0ab8c14d98498276b479255e128b.js"></script>

**We train** the model throughout a relatively small number of epochs of 7 since our input data are already pre-trained and will overfit after too much epochs, also because the batch size of 32 works with a large number of inputs due to our large data set. Note that you have to train reviews **x** against labels **y** (in this case the 5-star ratings).

<script src="https://gist.github.com/kraenkem/b0f0bfbb1efdeec7d808b69beb521d0e.js"></script>
<script src="https://gist.github.com/kraenkem/e29cde258cb852f791516ef34cdc5775.js"></script>
<script src="https://gist.github.com/kraenkem/f54e681b995cefce72b641a12c50a88c.js"></script>

<br>
**Model evaluation** is with 69 % quite high how a comparison with the results from Yang et al. as well as from others shows (see table below).
**Also, history plots** show that the training data set perform pretty well. Still, this is unfortunately not supported by the validation data set. This might be because of the small number of words we could proceed after the embedding layer which filtered out almost 70 % of all tokens due to misspelling. This might be improved with an even smaller batch and epoch size, or with a better, less mistaken data set.

<img align="center" width="700" height="500"
     style="display:block;margin:0 auto;"  
	 src="/blog/img/seminar/HAN_img/doc_class_comp.JPG">
Note: HN-AVE and -MAX refers to hierarchical network with averaging and max-pooling methods, HN-ATT refers to hierarchical attention network described in this blog.
<br>
<br>

## News Classification

To further demonstrate the attention mechanism, we also implemented the HAN on news articles to be able to classify them into categories, as well as to gain short summaries of articles by extracting the most important sentences using sentence attention weights. News articles are particularly interesting within this context because the hierarchical architecture can fully exploit the length of the document, compared to shorter documents, such as tweets.
We used a publicly available dataset from the British Broadcasting Corporation (BBC) which contains 2225 news articles from the BBC news website from 2004-2005. The news articles are sorted after five different categories: business, entertainment, politics, sport, and tech. 

### Parameters
As news articles tend to be longer than product reviews on average, we adjusted the parameters and increased the maximum number of sentences in one document and the maximum number of words in each sentence.

<script src="https://gist.github.com/leeh1234/1c7888cc8ca298ca0a2e071df8c761dc.js"></script>

### HAN Model

<script src="https://gist.github.com/leeh1234/5b75b55aa6e5328a7ab996bf392ae689.js"></script>

<br>
<br>
Our testset shows the following results:
<br>
<script src="https://gist.github.com/leeh1234/0cbcce6a9cbdcaf3122951cd2d522b63.js"></script>


<img align="left" width="340" height="280"
     style="display:table;margin:0 auto;"  
	 src="/blog/img/seminar/HAN_img/output_43_1.png">
	 
<img align="right" width="340" height="280"
     style="display:table;margin:0 auto;"  
	 src="/blog/img/seminar/HAN_img/output_43_2.png">

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
Compared to the Amazon dataset, the BBC dataset exhibits a much higher accuracy rate. This is probably due to several facts: news articles are in general much longer than product reviews, and therefore the HAN can exploit this and gain more information. Also, news articles have no grammar and spelling mistakes, while product reviews written by users just burst from them. Grammar and spelling mistakes lead to misinterpretation of words and thus loss of information. Another aspect is that the categorization classes of the BBC dataset are much easier to distinguish, whereas the star rating categorization of Amazon are very subjective and it is quite hard to draw a straight line between different categories.   

### Input new articles
To access newly released articles from BBC, we need to scrape the BBC website and save the title and text which is then cleaned, as described in the preprocessing, and subsequently converted to a sequence of numbers. 
<script src="https://gist.github.com/leeh1234/3ecc73e0f3c2c163c8e0dea73f33981e.js"></script>

### Sentence Attention Model
Now, we need to build a new model to be able to extract the attention weights for each sentence. This is to identify the five most important sentences within a news article to put them together and create a short summary. 
<script src="https://gist.github.com/leeh1234/98553337afe357407002c5a698ac8a46.js"></script>

### Word Attention Model
Additionally, we want to extract the usually hidden word attention weights as well for which we need to build another model. The words with the most attention serve as a good overview or framework for the article. 
<script src="https://gist.github.com/leeh1234/6b5d21606a399d01fc1d5aca0d076469.js"></script>
<script src="https://gist.github.com/leeh1234/ed879dc80e2d904c9cd7a33ae4c56a9f.js"></script>



<script src="https://gist.github.com/leeh1234/07ac108692fe2afd5096a899e6ac4c25.js"></script>
<br>
<br>


Words with the
most attention are used as new tags database can be created with taggs, summarized news articles

<script src="https://gist.github.com/leeh1234/13b13ff87bbc6f30e6c6669d3f9b063e.js"></script>

<br>
<br>

## Take Away

As you can see, hierarchical attention network is a well performing instrument for text classification. We hope this blog post - regardless of its mass of information - gives you understanding of how to use HAN. The most relevant points to remember are: 

* the hierarchical structure of documents (document - sentence - word),
* paying attention on contexts of sentences and words,
* by considering changing contexts, HAN performs better for classification problems.
