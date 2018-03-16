+++
title = "Wide and Deep Learning Model for Grocery Product Recommendations"
date = '2018-03-15'
tags = [ "Deep Learning", "Recommendation", "Class17/18",]
categories = ["seminar"]
banner = "img/seminar/recommendation/banner.jpg"
author = "Class of Winter Term 2017 / 2018"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = "Exploring and applying current trends in machine learning to a large scale product recommendation based on implicit feedback."
+++


**Recommendation Systems**
=============
<img align="center" width="800" height="400" src="/blog/img/seminar/recommendation/banner.jpg">



<br>
## **Introduction**

The explosion of information with the advent of the Internet and the multitude of choices available to customers introduces complexity in a customer’s decision processes. Recommender system is a useful information-filtering tool, which guides customers to a narrower selection of products and consequently helps them make better decisions. Matching users to the right products saves customers time and effort leading to increased user satisfaction, which in turn earns customer loyalty.

Personalization for a customer is evident in Amazon’s “Customers Who Bought This Item Also Bought” section. Such techniques are critical tools to promote sales and services in many websites and mobile applications. For example, the Netflix recommender system influences the choice for [80%](https://dl.acm.org/citation.cfm?id=2843948) of the hours streamed and [60%](https://dl.acm.org/citation.cfm?doid=1864708.1864770) of the video clicks come from the home-page recommendation in YouTube.

A personalized recommender system models a user’s preferences based on the interactions with different items/products. Among the various collaborative filtering techniques that do this, matrix factorization (MF) is the most popular. Although it is a successful technique, the drawback is that it relies on a dot product of very sparse vectors. Not only does this restrict scalability, it might also prove to be insufficient to learn the interaction between users and products. As a compensation, bias terms are added but this fails to sufficiently capture the complex interaction function as well.

In the past few decades, deep learning has been tremendously successful in a wide range of applications and has shown [state-of-the-art](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) results in recommender architectures as well. Due to its ability to learn complex patterns, neural networks architectures can be more effective in learning the interaction function.

While compelling recommendation systems are already in place for movies, songs, news and video recommendations, in this paper, we chose a successful large-scale architecture for recommending products and apply it to e-commerce data. Unlike explicit feedback like ratings and reviews available for movies, e-commerce data and online grocery shopping data in particular has to rely on implicit feedback. Although it is easier to collect this feedback in terms of clicks, purchases and time spent viewing products, implicit feedback is asymmetric and thus more difficult to utilize. It now becomes important to consider all the interactions a user has with the product instead of optimizing the model according to a limited set of features.

Simple machine learning techniques like linear models work well in understanding product-user co-occurrences and exploit the correlation generated from the historical interactions. On the other hand, advanced methods such as deep learning are able to explore new product-item co-occurrences that have never (or rarely) occurred before. The [Wide and Deep learning framework](https://arxiv.org/abs/1606.07792) for a recommender system combines the positive traits of linear models and deep learning. Due to a combinational network approach, this framework can learn all the patterns of user behavior from the additional information generated from feature engineering.

The work presented in this paper models the interaction function between users and items using a Multilayer Perceptron (MLP) and compares it to the simple matrix factorization approach, to illustrate the advantages deep learning has to offer in the recommender application setting. Further we build the Wide and Deep learning model and test it on a publicly available grocery shopping data.


<br>
<br>
## **Methods**

<br>
#### Matrix Factorization

Let M and N denote the number of users and items, respectively. We define the user-item interaction matrix as <img src="http://chart.googleapis.com/chart?cht=tx&chl=$Y \in \mathbb{R}^{MN}$" style="border:none;"> from users implicit feedback of reorder frequency, which is an integer. The recommendation problem of learning from implicit feedback can be
abstracted as learning <img src="http://chart.googleapis.com/chart?cht=tx&chl=$\hat{y_ui} = f(u, i | \Theta)$" style="border:none;"> where <img src="http://chart.googleapis.com/chart?cht=tx&chl=$\hat{y_ui}$" style="border:none;"> denotes the predicted score of interaction <img src="http://chart.googleapis.com/chart?cht=tx&chl=$y_ui$" style="border:none;">, <img src="http://chart.googleapis.com/chart?cht=tx&chl=$\Theta$" style="border:none;"> denotes model parameters, and <img src="http://chart.googleapis.com/chart?cht=tx&chl=$f$" style="border:none;"> denotes the function that maps model parameters to the predicted score.

MF associates the user and item with a real-valued vector of latent features.
Let <img src="http://chart.googleapis.com/chart?cht=tx&chl=$p_u$" style="border:none;"> and <img src="http://chart.googleapis.com/chart?cht=tx&chl=$q_i$" style="border:none;"> denote the latent vector for user <img src="http://chart.googleapis.com/chart?cht=tx&chl=$u$" style="border:none;"> and item <img src="http://chart.googleapis.com/chart?cht=tx&chl=$i$" style="border:none;">, respectively; MF estimates an interaction <img src="http://chart.googleapis.com/chart?cht=tx&chl=$y_{ui}$" style="border:none;"> as the inner product of <img src="http://chart.googleapis.com/chart?cht=tx&chl=$p_u$" style="border:none;"> and <img src="http://chart.googleapis.com/chart?cht=tx&chl=$q_i$" style="border:none;"> :
<img src="http://chart.googleapis.com/chart?cht=tx&chl=$\hat{y_{ui}} = f(u, i | p_u , q_i ) = p^T_u q_i = \sum_{k=1}^{K} p_{uk}q_{ik}$" style="border:none;"> where <img src="http://chart.googleapis.com/chart?cht=tx&chl=$K$" style="border:none;"> denotes the dimension of the latent space.

As the user ID and item ID are embedded, they can be seen as latent vectors which are linearly combined through a dot product operation. The weights for these embeddings are then optimized to learn the most effective representations. We also add a bias term by including an embedding of one dimension for the user and the item.

A basic factorization algorithm can be denoted as <img src="http://chart.googleapis.com/chart?cht=tx&chl=$(p_u q_i) %2B b_u %2B b_i$" style="border:none;">. This computes the implicit feedback for one (user, item) pair. It is evident from the formula that it can be interpreted as a logistic regression which multiplies a feature vector with weights and then adds a bias. While <img src="http://chart.googleapis.com/chart?cht=tx&chl=$(p_u*q_i)$" style="border:none;"> models the interaction between the user and the item, the bias terms model the effect of one dimension on the other. For example, in the Netflix challenge example, the bias of a movie would describe how well this movie is rated compared to the average, across all movies. This depends only on the movie (as a first approximation) and does not take into account the interaction between an user and the movie.

Consider a user who is new. In this case, <img src="http://chart.googleapis.com/chart?cht=tx&chl=$p_u$" style="border:none;"> denotes the factor for the user and would reduce to <img src="http://chart.googleapis.com/chart?cht=tx&chl=$0$" style="border:none;">. This means that your <img src="http://chart.googleapis.com/chart?cht=tx&chl=$p_u q_i$" style="border:none;"> will not predict any items for this user. The bias term ensures that this user is at least given the item chosen on average.

<script src="https://gist.github.com/brijda/8e8cbe2b6a2323dfdb614f3c041b7064.js"></script>



<br>
#### Multi-Layer Perceptron

The latent vector of the user and item are created through embedding the ID into a dense vector representation. These latent vectors are concatenated but this is not sufficient to model interactions between the users and items. Dense layers of neurons are added to sufficiently model the collaborative filtering effect.

The MLP is designed in a pyramid fashion with the bottom hidden layer having the maximum number of neurons. Each successive layer has half the neurons as the previous layer, making the last hidden layer the smallest. By reducing the size of the hidden layers closer to the output layer, the model can be forced to learn more abstractions from the data.

The MLP model can be mathematically synthesized as:

![](/blog/img/seminar/recommendation/mlp.PNG)

where <img src="http://chart.googleapis.com/chart?cht=tx&chl=$W_x$" style="border:none;">, <img src="http://chart.googleapis.com/chart?cht=tx&chl=$b_x$" style="border:none;">, and <img src="http://chart.googleapis.com/chart?cht=tx&chl=$a_x$" style="border:none;"> denote the weight matrix, bias vector, and activation function for the <img src="http://chart.googleapis.com/chart?cht=tx&chl=$x$" style="border:none;">-th layer's perceptron respectively.

The hidden layers introduce non linearity through the ReLU activation. Although sigmoid and tanh can also be used as the activation function to introduce non-linearities, they have certain limitations. The sigmoid function restricts the output to be in the range <img src="http://chart.googleapis.com/chart?cht=tx&chl=$(0,1)$" style="border:none;"> and it will cause the model to suffer from saturation if the neurons in the hidden layers are close to <img src="http://chart.googleapis.com/chart?cht=tx&chl=$0$" style="border:none;"> or <img src="http://chart.googleapis.com/chart?cht=tx&chl=$1$" style="border:none;">. With a range constraint of <img src="http://chart.googleapis.com/chart?cht=tx&chl=$(-1,1)$" style="border:none;">, tanh function introduces the same problem. As a result, ReLU was the best option for non-saturation. It has also been proved to work well with sparse data and prevent overfitting.

<script src="https://gist.github.com/brijda/f32275f435a2d720d4c5e11fb1cb1cc1.js"></script>



<br>
#### Wide and Deep Model

![](/blog/img/seminar/recommendation/widedeep.png)

The wide and deep learning has two individual components. The wide network is a linear estimator or a single layer feed-forward network. By assigning weights to each features and adding them with a bias term, it models the matrix factorization method. The deep neural network learns better representations of the latent vectors and introduces non-linearities in the latent representations for user and item. By jointly training the wide and deep network, the weights are optimized by back propagating the gradients
from the output to each network simultaneously.

The wide learning is defined as: <img src="http://chart.googleapis.com/chart?cht=tx&chl=$y = w^Tx %2B b$" style="border:none;"> as shown in above figure (left) where <img src="http://chart.googleapis.com/chart?cht=tx&chl=$y$" style="border:none;"> is the prediction, <img src="http://chart.googleapis.com/chart?cht=tx&chl=$x$" style="border:none;"> is the input feature vectors, <img src="http://chart.googleapis.com/chart?cht=tx&chl=$w$" style="border:none;"> are the model parameters and <img src="http://chart.googleapis.com/chart?cht=tx&chl=$b$" style="border:none;"> is the bias.

In the deep component of the network, each layer is in the form of <img src="http://chart.googleapis.com/chart?cht=tx&chl=$a^{l%2B1} = f(W^{(l)} a^{(l)} %2B b^{(l)} )$" style="border:none;">, where <img src="http://chart.googleapis.com/chart?cht=tx&chl=$l$" style="border:none;"> indicates the <img src="http://chart.googleapis.com/chart?cht=tx&chl=$l$" style="border:none;">th layer, <img src="http://chart.googleapis.com/chart?cht=tx&chl=$f (\cdot)$" style="border:none;"> is the activation function, <img src="http://chart.googleapis.com/chart?cht=tx&chl=$W^{(l)}$" style="border:none;"> is the model weights and <img src="http://chart.googleapis.com/chart?cht=tx&chl=$b^{(l)}$" style="border:none;"> is the bias term.

The wide & deep learning model is obtained by fusing these two models:

![](/blog/img/seminar/recommendation/wd.PNG)

where <img src="http://chart.googleapis.com/chart?cht=tx&chl=$\sigma(\cdot)$" style="border:none;"> is the sigmoid function, <img src="http://chart.googleapis.com/chart?cht=tx&chl=$\hat{y_{ui}} $" style="border:none;"> is the binary rating label of whether a user reordered a particular item, <img src="http://chart.googleapis.com/chart?cht=tx&chl=$a^{(l_f )}$" style="border:none;"> is the layer activation.

Feature engineering is an important step in selecting the inputs to the wide and deep components of the model. Attributes with continuous data and categorical columns are included as the input to the wide network.
<script src="https://gist.github.com/brijda/2029d6608023fd5886e9a2c96250fdbb.js"></script>

The wide model is trained on binary sparse features with one-hot encoding.
<script src="https://gist.github.com/brijda/e5b6bd11b6e3abbf129fb6a09d09c2de.js"></script>

In addition to these features, the user and item IDs which are to be embedded as dense vector representations, are fed to the deep neural network.
<script src="https://gist.github.com/brijda/3cf441b120becd622f20c3f23257830a.js"></script>

The embedding vectors are initialized randomly and during model training, these embeddings are also optimized to minimize the final objective function.

The wide model memorizes the user-product interactions present in the data. So the recommendations based on the wide model are usually more topical.
<script src="https://gist.github.com/brijda/6af3c88139b771d3d1817ee003f8b1ff.js"></script>

Apart from the feature columns present in the data collected during user interactions with products, cross-column features are created as additional side information. Aggregate statistics are computed for each user by using the meta-data corresponding to that users interaction with different items. Similarly, product based features are generated as well. The deep model is a feed forward neural network as shown in figure (right).
<script src="https://gist.github.com/brijda/8f644f5e4a4267ca0f0f7e2cd11a7187.js"></script>

The combined wide and deep model, as shown in figure (middle), takes the weighted sum of the outputs from both components as the prediction value. This prediction is fed to one common logistic loss function for joint training.  
<script src="https://gist.github.com/brijda/e75c8180114f36f572f29b959fb18dd0.js"></script>



<br>
#### Ranking Methods to recommend products

Ranking, or to recommend a set of products does not required the complicated algorithms compared to the previous predictive models. Since we already predicted the probability if a user will reorder a certain item in current order, the rank of the products in this order will be used for the recommendation.

There are two approaches to recommend products based on reorder probabilities:
First, set a threshold for the probabilities, if the predicted probability is above the threshold, this product will be included in the recommend set for this order.Second, set the number of products we want to recommend, if we want recommend 5 products for this order, we take the products from top 5 highest reorder probabilities we predict for the recommend set.

The advantage of first approach is, it can be evaluated with previous order and thus find the best threshold using a simple F1 score maximization. In another word, apart from the predictive model we can still improve the recommendation performance. The down side is, however, in some scenarios there might be too much products in the recommendation set. The second approach can make sure users get a right amount of products, which is more controllable, while the performance can only be improved from predictive models.



<br>
## **Experiments**

In this section, we conduct experiments with the aim of answering the following research questions (RQs):

  • **RQ1** Do the deep learning based models outperform collaborative filtering techniques?

  • **RQ2** Does the number of dimensions chosen for the embeddings of users and items impact performance of the model?

  • **RQ3** Are more layers of hidden neurons better for the deep learning models?

  • **RQ4** Does feature engineering improve the predictive power of recommender models?

We first present the experimental settings followed by the outcomes of the research questions.



<br>
#### Experimental Settings

**Dataset**: We experimented with a publicly available dataset for grocery data. The Instacart dataset accessible through Kaggle has around 3 million grocery orders of about 200,000 users. For each user, 4 to 100 orders and the sequence of their orders is available. Some other attributes which were collected are time of the day and day of the week an order was placed.

**Baselines**: Matrix factorization is compared to a MLP. The wide model and deep model are also individually compared with the combined performance of the jointly trained wide and deep model.

**Parameter Settings**: The methods developed and compared in this article are based on Keras.



<br>
#### RQ1 • Performance comparison

To compare the matrix factorization method with a MLP, a user versus product matrix is utilized with the implicit feedback of number of times a product has been reordered by the user.  An MLP with three hidden layers of 200,100 and 50 neurons is compared with the MF model. The latent vector for user ID and item ID is kept the same for both the models. Over different sizes of the datasets, the MLP model consistently outperforms the predictions of the matrix factorization technique. The results of both the models are shown in the table below.

![](/blog/img/seminar/recommendation/rq1 table.PNG)

The wide and deep neural network, as expected, outperforms the baseline models. While the linear estimator learns better than the deep neural network, the combined performance of both is more impressive than the
individual models. The latent vectors were set to be of length 8. The comparison of the models can be seen in the table below.

![](/blog/img/seminar/recommendation/rq1 table1.PNG)


<br>
#### RQ2 • Length of latent vectors

The embedding vectors for user ID and product ID were varied. Different lengths of 20, 50 and 100 were tested to note the differences in performance of the wide and deep network. These results are tabulated below.

![](/blog/img/seminar/recommendation/rq2 table.JPG)

Loss doubles with the increase of embedding length, the other metrics are basically the same. There may be some problems with the AUC of deep  model, no matter how embedding length changes, it’s always a perfect 0.5. Increasing embedding length to 50 from 20 did increase the accuracy of by 1.4% and 2% compare to length of 8. There is also a slightly increase by AUC, but AUC also decrease when embedding length changes from 8 to 20.

![](/blog/img/seminar/recommendation/rq2 table.JPG)
![](/blog/img/seminar/recommendation/rq2 acc.png)
![](/blog/img/seminar/recommendation/rq2 auc.png)
![](/blog/img/seminar/recommendation/rq2 f1.png)
![](/blog/img/seminar/recommendation/rq2 loss.png)
![](/blog/img/seminar/recommendation/rq2 logloss.png)




<br>
#### RQ3 • Depth of hidden layers

We set out to check whether the number of hidden layers and the number of neurons per hidden layer can help us fine tune our deep model for better accuracy. First we check the impact of adding one hidden layer to the model. The table below shows the accuracy and loss changes with variation in hidden layers and their size across multiple iterations for our deep model without additional features.

![](/blog/img/seminar/recommendation/rq3 table.png)

There is a clear distinction between accuracy of two layered models and three layered models. The mean accuracy for two layered deep models is a little over 0.65 whereas the mean accuracy for three layered models is about 0.59. So learning the negative impact of increasing number of hidden layers, we perform more iterations for two layered deep model.
The graph below shows the variation in accuracy on primary vertical axis and loss on secondary vertical axis with varying number of neurons in two layered deep model.

<img align="center" width="600" height="300" src="/blog/img/seminar/recommendation/rq3 graph1.png">

We could not find a definite positive or negative impact with increasing or decreasing the number of neurons. As shown, higher accuracy is achieved by (100, 80), (60, 20), (50, 20) and (40, 15). So we can say that increasing the number of neurons does not necessarily increase accuracy. On some further analysis, we suspect the ratio of number of neurons in the two layers does likely have an impact on accuracy. The following graph shows the accuracy variation against the ratio of the two hidden layers sizes. We find that accuracy peaks at the neuron ratio of about 2.5.

<img align="center" width="500" height="300" src="/blog/img/seminar/recommendation/rq3 graph2.png">




<br>
#### RQ4 • Importance of feature engineering

Here we compare the results of the original model with additional feature engineering to a model without the new features. The table below shows the accuracy and loss for the Wide, Deep and Wide & Deep models. There is a clear impact of the additional feature engineering effort. The outperforming wide model improves accuracy by about 18% and the deep model by 26%. The losses also show some improvement after implementing the new features.

![](/blog/img/seminar/recommendation/rq4 table.png)






<br>
## **Conclusion & future work**

In this work, we explored neural network architectures for collaborative filtering. We compared a multilayer perceptron with matrix factorization technique and show that neural frameworks are better at learning the user-item interactions. A Wide & Deep Learning framework is developed and evaluated on grocery basket data. Effect of model performance was studied by varying the latent vector lengths. While adding hidden layers did not improve the performance of the deep model, selection of appropriate number of neurons in each layer boosted the model accuracy. Finally we also saw the huge positive impact of practicing feature engineering.

Possible extension to the wide and deep learning model:The sequence in which a user adds the products to the basket is a critical feature. During each addition of a product into the basket, recommending the next few products in the pattern in which a user adds them into the basket can improve the success of the recommender system. An LSTM which is good at learning sequence based patterns can be applied as the deep learning model to utilize this feature.

Baseline Comparison: The models can be compared with the state-of-art collaborative filtering techniques to assess if they outperform traditional recommender algorithms. [Bayesian Personalized Ranking](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) is a highly competitive baseline for item recommendation as it combines matrix factorization with pairwise ranking loss which is tailored to learn from implicit data.
