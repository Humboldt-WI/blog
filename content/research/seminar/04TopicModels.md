+++
title = "Numeric representation of text documents: doc2vec how it works and how you implement it"
date = '2018-03-15'
tags = [ "Deep Learning", "Text Mining", "doc2vec", "Document Embeddings", "Class17/18",]
categories = ["Course projects"]
banner = "img/seminar/topic_models/textMininigKlein.png"
author = "Seminar Information Systems (WS17/18)"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = "Candidate2vec - a deep dive into word embeddings"
+++

# Numeric representation of text documents: doc2vec how it works and how you implement it

#### Authors: Felix Idelberger, Alisa Kolesnikova, Jonathan Mühlenpfordt

# Introduction

Natural language processing (NLP) received a lot of attention from academia and industry over the recent decade, benefiting from the introduction of new algorithms for processing the vast corpora of digitized text. A set of language modeling and feature learning techniques called word embeddings became increasingly popular for NLP tasks. Word2vec (Mikolov et al, 2013) became one of the most famous algorithms for word embeddings, offering a numeric representations of any word, followed by doc2vec (Le et al, 2014), which performed the same task for a paragraph or document.

The area of application is wide and encompasses online advertisement, automated translation, voice recognition, sentiment analysis, topic modeling and dialog agents like chat bots. Academia also received an efficient tool allowing to tap into the vast corpus of digital texts, from Shakespeare works to social media.

<img style=" width:100%;display:block;margin:0 auto;"
src="/blog/img/seminar/topic_models/vectors.png">


Regardless of the specific task at hand, its overarching goal is to obtain numeric representation of a document that reflects its content and, possibly, some latent semantic relationships within.New machine learning algorithms are often not well documented or miss in-depth explanation of the processes under the hood. One of the possible explanations suggests that  many were developed by teams from technology companies like Facebook or Google, that would rather focus on application aspect. Doc2vec is no exception in this regard, however we believe that thorough understanding of the method is crucial for evaluation of results and comparison with other methods.

This blog post is dedicated to explaining the underlying processes of doc2vec algorithm using an empirical example of facebook posts of German political candidates, gathered during the year before elections to Bundestag.


<img style=" width:100%;display:block;margin:0 auto;"
src="/blog/img/seminar/topic_models/oprah.png">

We will start by providing some __desciptive statistics__ of the dataset, followed with some __theoretical backgound__ to embedding algorithms, in particular, word2vec's __SkipGram__ and doc2vec's __PV-DBOW__, moving on to step-by-step __implemetation__ in python, followed by the same actions performed using the gensim package. We will finish this blog by showing the __application__ of the discussed methods on the political data, also touching __vizualisation__ techniques.

# Preparatory steps

In preparation for the following steps, we need to specify which packages to load and where we have stored the data:

<script src="https://gist.github.com/jgmill/0dd8e0691df5622869c53c5da2f38159.js"></script>

# Descriptive Statistics

The dataset consists of 177.307 Facebook posts from 1008 candidates and 7 major political parties who ran for the German Bundestag in the 2017 election. We collected all messages posted between 1 January and the election date on 24 September 2017, covering the entire campaigning period. The parties are included in order to later compare the candidates to them, but computationally they are treated no different then the candidates.

Sahra Wagenknecht, with a mean of 99.63 words per post, publishes quite long posts compared to Joachim Herrmann and Cem Özdemir who use on average less than half the amount of words in their posts.

<script src="https://gist.github.com/jgmill/2cccc7c1752f862b7ce3cd3951e36464.js"></script>

Words per post by party: AfD and Die Linke candidates write longer posts on average than the candidates from the other parties.

<script src="https://gist.github.com/jgmill/1ef09fd002e985e494ccf04599234091.js"></script>

A histogram of the words per post across all parties: A large majority of posts has less than 50 words

<script src="https://gist.github.com/jgmill/d58de0607d4eed89f61af1f1059288ad.js"></script>


# Theory


The goal of the empirical case study is to generate a concept space for each candidate, thus offering a visual representation of German political landscape. For this purpose we need an algorithm to construct an embedding layer that will contain the numeric representation of the semantic content of each candidate's Facebook posts.

PV-DBOW is similar to the Skip-gram model for word vectors (Mikolov et al, 2013), we will revisit the word2vec algorithms to facilitate the methodological transition.

<img style=" width:100%;display:block;margin:0 auto;"
src="/blog/img/seminar/topic_models/skipgram.png">

Word2vec is built around the notions of "context" and "target". While CBOW predicts a target word from a given context, Skip-gram predicts the context around a given word. Using an example from our dataset, we would try to get "teilweise", "laute", "Gesetzentwurf" by giving "Diskussion" to the network (originally "teilweise laute Diskussion um den Gesetzentwurf").

PV-DBOW performs a very similar task with the only difference of supplying a paragraph id instead of the target word and training a paragraph vector. Unlike PV-DM, PV-DBOW only derives embeddings for documents and does not store word embeddings.

<img style="display:block;margin:0 auto;"
src="/blog/img/seminar/topic_models/pvdbow.png">

Let's try to train embedding vectors for each candidate in our dataset, using his or her facebook posts as one big text document.

**Some preparation**

We combined all the posts issued by the same candidate/party in order to obtain one document per candidate. By doing this, we ended up with 1015 text documents (1008 polititians and 7 parties) each containing the rhetorics of one candidate during campaigning.These paragraphs became the "docs" in our implementation of doc2vec and led us to giving the project the "candidate2vec" nickname. Each document is further tokenized in order to allow distinguishing individual words. Finally, we filter out stop words of the German language.

<script src="https://gist.github.com/jgmill/4ce0cab852d0e599a5f700af2a998c37.js"></script>

**Some more preparation**

A few more preparatory steps are required before the actual training: As a first step, a list of the __m__ unique words appearing in the set of documents has to be compiled. This is called the  vocabulary. Similarly, a list of of the documents is needed. In our case, these are the __n__ aggregated corpora of Facebook posts of every candidate and the political parties.

For every training iteration, one document is sampled from the corpus and from that document, a word window or context is selected randomly.

<script src="https://gist.github.com/jgmill/b84e33ac74de8bae4dc369b5ace27794.js"></script>

In order to explain the internal structure, we need to get a clear understanding of the PV-DBOW workflow:

<img src="/blog/img/seminar/topic_models/networkstructure.png">


**Input Layer**

The input vector __d__ is a one-hot encoded list of paragraph IDs of length __n__, with a node turning into 1 when the supplied ID matches the selected document.

**Hidden layer**

The hidden layer consists of a dense matrix __D__ that projects the the document vector __d__ into the embedding space of dimension __p__, which we set to 100. __D__ thus has dimensions __p__x__n__ and is initialized randomly in the the beginning before any training has taken place.

After the training, __D__ will constitute a lookup matrix for the candidates, containing weights for the paragraph vector __e__ (standing for embeddings). The latter will contain the estimated features of every document.

**Output layer**

In the output layer, the matrix __U__ projects the paragraph vector __e__ to the output activation vector __k__ which contains __m__ rows, representing the words in vocabulary. __U__ has dimension __m__x__p__ and is initialized randomly similar to __D__.

<script src="https://gist.github.com/jgmill/d5d8fdf62d350cc634fd87da7cd0bca3.js"></script>

**Softmax**

The process of predicting the words from the context given the supplied paragraph ID is done through a multiclass classifier called softmax, which provides a probability distribution over the words for an input document.

__k__ is then passed to the softmax function in order to obtain __t__^, the vector of the predicted probabilities of each word in the vocabulary to appear in the document

<img style=" width:20%;display:block;margin:0 auto;" src="/blog/img/seminar/topic_models/softmax.png">

**Backpropagation and cross-entropy**

In the last step, the softmax probabilities __t^__ are compared with the actual words from the selected context. We can compute the cross entropy loss function, which sums up the products of the component-wise logarithm of __t^__ and the actual one-hot encoded vectors __t__. Since __t__ is zero for the words outside the context, only the prediction error for the context word is taken into account.

<img style=" width:25%;display:block;margin:0 auto;"
src="/blog/img/seminar/topic_models/lossfunction.png">

The goal of the SGD is then to gradually reduce the loss function, minimizing the resulting difference with every iteration.

The gradients of the loss function on the  matrices __D__ and __U__ will tell us how we need to update the weights that connect the neurons in order to reduce the loss function. They are obtained by taking the derivative of __E__ with regard to the components of matrices __D__ and __U__. For Skip-gram, \textcite{rong_word2vec_2014} shows how we need to update the weights by passing the prediction error back through the network, which cab be adapted to PV-DBOW as well. We therefore compute the output prediction error __o__ of length __m__ by substracting __t__ from __t^__ and the error from the embedding layer __h__ by projecting __o__ into the embedding space by means of right-multiplying it onto __U'__.

The errors are computed for each of the __c__ words in the given context and summed up before passing them back through the network. The sum of the output errors is right-multiplied with the transposed embedding vector __e'__ in order to obtain the update values for __U__, while the sum of the embedding-layer errors gets right-multiplied by __d'__. Before updating __U__ and __D__, the updates are multiplied with learning rate __alpha__ which limits how much the weights are adjusted at each step. A common choice is to start with __alpha=0.025__, gradually reducing it to __0.001__.

The backpropagation concludes the training iteration for one document. Repeating these steps for all documents once would then constitute one epoch.

<script src="https://gist.github.com/jgmill/e6296db5df46394a23c15f7e8426fd06.js"></script>

**A note on efficiency**

As running softmax on the whole vocabulary would be computationally inefficient, the following techniques are usually considered: hierarchical softmax (indexing the words in vocab) and negative sampling (output layer will contain correct words and only a bunch of incorrect ones to compare to).

**Putting everythin in one loop**

Here, we combine all the steps previously explained in order to complete the training for 1 epoch (or more if you want)

<script src="https://gist.github.com/jgmill/8aedbef371e7ae5d22881612a2f64140.js"></script>

**Visualisation with t-SNE**

The paragraph vectors contain 100 components, which makes them hard to visualize and compare. A t-Distributed Stochastic Neighbor Embedding (t-SNE)(Maaten, 2008) is a popular technique for dimensionality reduction that is used widely in NLP. The concept follows the paradigm of visualizing similarities as proximity by minimizing an objective function that signifies discrepancies in the initial multidimensional format and final visual representation. Retention of information remains an obvious challenge when reducing the dimensions. t-SNE manages to preserve the clustering of similar components (unlike Principal Component Analysis, which focuses on preserving dissimilar components far apart). The algorithm assigns a probability-based similarity score to every point in high dimensional space, then performs a similar measurement in low dimensional space. Discrepancies between the actual and simplified representations are reflected in different probability scores and get minimized by applying SGD to the Kullback-Leibler divergence function, containing both scores (Kullback, 1951). The "t" in the name refers to the student distribution that is used instead of the Gaussian distribution when estimating probability scores, as the thicker tails allow to maintain larger distance between the dissimilar points during the score assignment, thus allowing to preserve local structure.

<script src="https://gist.github.com/jgmill/43fecbedb42be8663f4e7a017656a75d.js"></script>

As we see, the simple model does not show any pattern among the candidates yet. In the following section, we use the python package gensim which allows run many more trining iterations in less time.

# Application of doc2vec in gensim

we apply **doc2vec** on facebook posts from politicans in the German election 2017

**GOAL:** to let the model find a distinctiveness in the data, e.g. a difference between party candidates

**Approach**:

- two models **PV-DBOW** and **PV-DM**
- 100 dimensions
- for 1, 5, 10, 20, 50 and 100 epochs
- ...

*Remember:*
one post = one doc

**Solution:**

- Gensim allows to tag documents. Our tag = name of the candidate
- model creates and trains vectors for the candidates

For clarity we omit some of the code here, but you can check out the full code [here](https://gist.github.com/panoptikum/d3023bc7619814fcea5235e0f472052c). Due to the copyrights of Facebook we cannot provide the data set here.

If you want to use the interactive plotly library in your jupyter notebook, you have to put the following lines at the beginning of your notebook:

<script src="https://gist.github.com/panoptikum/b534310edb163cf48e5681d26b1ee8e3.js"></script>

Some other libraries are needed as well:

<script src="https://gist.github.com/panoptikum/e3a6774155c2e93e64f253d5faf2e1ad.js"></script>

we set our working, data and model directory:

<script src="https://gist.github.com/panoptikum/31d6fbf4ea29fd80c307812dcd7d041e.js"></script>

we load the data that we've cleaned a bit before (e.g. removal of facebook posts without text):

<script src="https://gist.github.com/panoptikum/6afdf5ce59b0b6b7ec1f6fe8c2766065.js"></script>

Party names are a quite distinctive pattern, so we remove them beforehand:

<script src="https://gist.github.com/panoptikum/cc4c3f5371a7c0fac944edf04dd687b5.js"></script>

this code chunk tokenizes and tags each posts with the candidate's name:

<script src="https://gist.github.com/panoptikum/17f05f9dfc26f48575540fda76e26ffd.js"></script>

doc2vec has a fast version which we activate with the following code:

<script src="https://gist.github.com/panoptikum/0dc7ca4687141448c1ca5480983ee25b.js"></script>

## doc2vec model

sets the parameters of the models:

<script src="https://gist.github.com/panoptikum/40cab8f6435e9b57998b9e312fd54950.js"></script>

builds the vocabulary once and transfer it to all models:

<script src="https://gist.github.com/panoptikum/f0d9a839ce1025bd5402430e411c795b.js"></script>

to be able to call the models by their name we use an ordered dictionary:

<script src="https://gist.github.com/panoptikum/cc038f4c7351ee12c27882a0318151d7.js"></script>

### training the models

here you can set the epochs of your choice. For debugging reasons We only ran one value of epochs at a time and not several:

<script src="https://gist.github.com/panoptikum/f41b2745cfc74986d2ae1542f8ce514e.js"></script>

We recommend to save the models for later use and if you test a set of parameters:

<script src="https://gist.github.com/panoptikum/6b8b40298c2f8b6afe4304872fb31d40.js"></script>

### a look on the resulting embeddings

To do some graphical analysis or to compute something with the model, we need candidate specific data such as party affiliation:

<script src="https://gist.github.com/panoptikum/103fe5b5085929e1881a2965d177291e.js"></script>

next we define some colors for the parties that we can identify the candidates in our graphs later on. Furthermore, we create an array that contains the party leaders that we can use a different shape for them. We do the same for the Party accounts:

<script src="https://gist.github.com/panoptikum/39fae3bdab4058d97085abe8765b5c43.js"></script>

#### plots

now we need the plotly graphic library and some other libraries. In addition, we load our trained models of PV-DM and PV-DBOW with 20 epochs:

<script src="https://gist.github.com/panoptikum/095f4daab1d97b5745722f5147e7bd0c.js"></script>

you can compare the models directly with each other, if you create a subplot. The following code chunks does this job, access the candidate embeddings (vector) reduces the dimensionality them to two dimensions and plots them:

<script src="https://gist.github.com/panoptikum/c7e3db70799548b1c33e5daab8ddafc3.js"></script>

let's finally look at the candidate embeddings (vectors) mapped into two dimensions:

<script src="https://gist.github.com/panoptikum/ca40fcf7c0da73c829ce6b58f9ce161f.js"></script>

In the graph below every point represents a candidate. You can hover over one to get the name of the candidate. Parties are drawn as squares and leader of parties as hexagons:

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~Stipe/122.embed"></iframe>

The following analysis is based on the PV-DBOW with 20 epochs well, but was equally applied to the 11 models as well.

#### Is a candidate most similiar to own or different party?

We are interested in whether a candidate is most similar to her own party or not according to the model. In order to determine this, we compute the cosine similarity between a candidate with each party and consider the party sharing the highest similarity with the candidate most similar. Be aware that the embeddings of the model may not present the actual semantic relationships between candidates and parties:

<script src="https://gist.github.com/panoptikum/342e67f72a0fa59c6dc8357d4b5bd6e9.js"></script>

Now we have a pandas data frame that contains the similarities of one candidate with all parties and a column that indicates which party is most similar:

<script src="https://gist.github.com/panoptikum/05d41932a8c71e888afa994e4482665f.js"></script>

##### How many candidates are most similar to their own party?

Let's do a quick crosstab in order to obtain a number of candidates for each possible pair:

<script src="https://gist.github.com/panoptikum/a9b5ee047fce3588f785c65a4540b7d0.js"></script>

##### What's the average similarity of candidates from one party?

Another interesting way to examine the models is to compute the average similarity of all candidates from one party to the semantic of their own party and to the other parties. We calculate this part with the following code chunk:

<script src="https://gist.github.com/panoptikum/7c6b893f1653d4730271c3c452219c4e.js"></script>

The resulting table looks as follows:

<script src="https://gist.github.com/panoptikum/7b7dc5ce0a88a8092f2ef0a6448a7131.js"></script>

##### Model comparison

Yes, we do not know the actual semantics of the politicians' Facebook posts, but we may assume that candidates use similar rhetorics like their party as a whole and share less rhetorics with other parties. If we compute the average of the diagonal elements and the average of the off-diagonal elements, we capture these two dimensions. The different of the the two means tells us how good the model captures the two dimensions:

<script src="https://gist.github.com/panoptikum/fca5cfbbfa8e5006c0c91633b36a63d4.js"></script>

We calculated this difference for all 12 models and we received the following results. You see that an increase in epochs do not change this metric significantly beyond 20 epochs:

<script src="https://gist.github.com/panoptikum/09c58895fdbf299f015c703776b5885c.js"></script>

# Playing around with the model

In reference to the famous king-queen example in the doc2vec literature, we test whether the models capture similar analogies. If we subtract 'Frau' from 'Bundskanzlerin' and add 'Mann' to it, we receive 'martinschulz' in the results. The similarity which the resulting vector shares with the word is rather weak:

<script src="https://gist.github.com/panoptikum/b169f1984641b19c72f637bd8af0a5aa.js"></script>

## similar words

Another nice feature is the fact that you can obtain the most similar word vectors for a word vector. We queried the model for several words that are shown next:

<script src="https://gist.github.com/panoptikum/0ff36e31f32d835131d909d37cbac09a.js"></script>

# The end

If you made it till here, you have digested a lot of information, well done! We hope that our introduction to doc2vec gives you a better understanding of how PV-DBOW works. We would be happy, if you feel encouraged to apply doc2vec to some data on your own.
