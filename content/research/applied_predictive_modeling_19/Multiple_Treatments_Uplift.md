+++
title = "Uplift Modelling with Multiple Treatments"
date = '2019-07-12'
tags = ["Causal Inference", "Class19","Uplift"]
categories = ["Course projects"]
banner = "img/seminar/sample/hu-logo.jpg"
author = "Seminar Applied Predictive Modeling (SS19)"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = "Evaluation and discussion of Uplift Models for multiple possible treatments in the area of Marketing Analytics."

+++


# Applications of Causal Inference for Marketing: Estimating Treatment Effects for multiple Treatments

#### Authors: Jan Krol and Matthias Becher

# Table of Contents

1. [Introduction](#introduction)<br />
2. [Common Marketing Challenges](#challenges)<br />
3. [Models](#models) </br>
3.1 [Decision Trees Rzepakowski & Jaroszewicz](#decisiontree) </br> 
3.1.1 [Basic Rzepakowski & Jaroszewicz](#basic) </br>
3.1.2 [Simple Splitting Criterion](#simple) </br>
3.2 [Causal Tree and Causal Forest](#causaltree) </br>
3.3 [Separate Model](#separate) <br />
4. [Evaluation Methods](#evaluationmethods)<br />
4.1 [Uplift Curves](#uplift_curves)</br>
4.2 [Expected Outcome](#expected_outcome)</br>
5. [Experimental Setup](#experiment) <br />
5.1 [Data Sets](#datasets)</br>
6. [Results](#evaluationresults)
7. [Outlook](#outlook)
8. [References](#references)

# 1. Introduction <a class="anchor" id="introduction"></a>
Nowadays marketing practitioners face a multitude of challenges. 

### TODO muss noch angepasst werden nach finaler Struktur
The following work is structured as follows.
In the first part of this analysis we examine existing challenges and how treatment effect analysis can be employed to tackle those challenges.

In the second part we look more closely at one specific problem and give a summary and an evaluation of proposed methods to solve that problem.

<br />

In the second part of this blog post we will look at methods which have been proposed so far to deal with multiple treatments.</br> 
In the first section we will give an overview and brief explanation of the methods we look at.</br>
In the second section we evaluation their performance both in terms of their predictions and in terms of the duration it takes to train the models. </br>
Finally, we will give a short summary of our results and a short outlook over possible future research.


# 2. Common Marketing Challenges <a class="anchor" id="challenges"></a>
Some of the most commonly cited challenges we found were:
<ul>
  <li>Drive Traffic to the website</li>
  <li>Increase/Prove ROI of marketing campaigns</li>
  <li>Turn website visitors into buyers</li>
  <li>Reach people at the right time</li>
  <li>Target content to the right audience</li>
  <li>Select marketing channel/method</li>
</ul>
Since those challenges were all related to marketing activity we decided to focus on how treatment effect anlysis could be used to improve the performance of marketing activities. More specificly there are 3 questions with regards to marketing activity we focused on.

### When?
When it comes to selecting the time of a marketing activity there is not much research out there with regards to treatment effects.
Most advice for marketing pratitioners focuses on social media marketing and on what time are best for publishing new posts. These guidelines mostly look at single performance measures like engagement to see at which times the performance measure are maximized. An example of this can be seen in the blog post 
<a  href="https://sproutsocial.com/insights/best-times-to-post-on-social-media/" target="_blank"> "Best times to post on social media for 2019"</a>. Generally these approaches only attempt to maximize the average treatment effect. Since social media usually works in a broadcast style in which each post reaches all users it is not possible to adjust the content or publishing time of a post for specific users or groups of users.</br>
In their paper <a href = "https://www.researchgate.net/publication/4753376_Time-Series_Models_in_Marketing" target="_blank"> Time Series Models in Marketing </a> the authors look at the application of time series models for marketing. For example they use persistance modeling in order to estimate  the longterm effect of marketing activities. The image below shows the long term impact of an activity on the price promotion elasticity. For detergent there is an immidiate effect, which levels off and reaches the original level after some time. For dairy creamer we also see some recution over time but it remains stable at an elevated level.

<figure>
<img
align="center"
width="305"
height="362"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/TimeSeries.PNG">
<figcaption align="center">
<a href="https://www.researchgate.net/publication/4753376_Time-Series_Models_in_Marketing">Source: Time-Series Models in Marketing</a>
</figcaption>
</figure>
</br>
Being able to estimate the long term effect of ones marketing activity allows the practicioner to select the appropriate starting point in order to maximize the ROI. </br>

Another approach to find seasonal effects might be to look at past marketing activities which have been similar in terms of the activity performed, but have been done at different times. Then one could estimate the treatment effects of each of these campaigns to get an idea at which time during the year the campaign works better. However, the activities should not be to far apart. Otherwise global factors like the state of the economy could have changed. This would also have an impact on the purchasing behavior of customers and could lead to false conclusions.</br>

### Who?
The importance of this question varies greatly depending on the kind of marketing that is being done. Figure 2 shows various types of marketing from broad to narrow. The narrower the more potential there is for the usage of treatment effects. For the broadest possible marketing activity (like the social media marketing mentioned before) the average treatment effect (ATE) is important but no selection cade be made in terms of who we target. Narrower activities might allow us to select certain subgroups of our potential customers. Here we would be interested in the group average treatment effects (GATES) of those subgroups. Then we could determine which group to target based on those treatment effects.

<figure>
<img
align="center"
width="300"
height="200"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Market-Targeting-Strategies.png">
<figcaption align="center">
<a href="https://marketing-insider.eu/wp-content/uploads/2015/04/Market-Targeting-Strategies.png">Source: https://www.marketing-insider.eu</a>
</figcaption>
</figure>

Our focus lies on the most narrow kind of marketing activities. This means activities like emails, coupons etc. which are specifically targeted towards the receiving person. Here we can decide on an individual basis whether we target a given potential customer and what treatment we use. This is in contrast to the broadest form, like social media posts, which reach all followers and not just specific ones. Historically practicioners would target the people who they thought would be most likely to do a purchase. This approach is suboptimal since it is solely based on the overall purchase possibility and not the effect of the treatment. </br>
In general we can separate our customers in 4 groups (Figure 3). </br>

<figcaption>
<img
align="center"
width="300"
height="200"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Mike-Thurber-Graphic-2.png">
<figcaption align="center">
<a href="https://www.predictiveanalyticsworld.com/patimes/wp-content/uploads/2017/03/Mike-Thurber-Graphic-2.png">Source: https://www.predictiveanalyticsworld.com</a>
</figcaption>
</figure>

With the historical approach we will target mostly the 'Sure Things' and maybe the 'Do-Not-Disturbs'. For those groups we at best get no return and at worst actually lose customers. Ideally we want to target the 'Persuadables'. This is commonly done by estimating the conditional average treatment effect (CATE) or uplift of the proposed marketing activity and then target the customers where the activity is estimated to have the highest effect. </br>
Several approaches have been proposed to estimate uplift. Gubela et. al (2019) give an overview in their paper <a href = "https://www.researchgate.net/publication/331791032_Conversion_uplift_in_E-commerce_A_systematic_benchmark_of_modeling_strategies" target="_blank"> Conversion uplift in E-commerce: A systematic benchmark of modeling strategies</a>. In their evaluation they find that the two model uplift method and interaction term method (ITM) performed best. </br>
The two model approach as the name suggests uses two separate models. The training set is split into two with one set contained all the treated observations and the other all control observations. Then for each traning set one model is built to predict the outcome. To estimate the uplift of the treatment for a new person, we generate the predicted outcome with both models. One predicted outcome with treatment and one without. The difference between these two estimates is the expected uplift. The two model is very simple and works with virtually every possible base model (e.g. random forest, linear regression, svm, ...). Since it is so simple it is often considered a good benchmark to test new approaches against.</br>
The interaction term method (ITM) was proposed by Lo in his 2002 paper <a href="https://www.researchgate.net/publication/220520042_The_True_Lift_Model_-_A_Novel_Data_Mining_Approach_to_Response_Modeling_in_Database_Marketing" target="_blank">The True Lift Model - A Novel Data Mining Approach to Response Modeling in Database Marketing</a>. Unlike double machine learning, ITM only uses a single model. However, ITM also works with two predictions. One with treatment and one without. Both predictions are obtained from the same model which has been trained on both treatment and control data. Whether a treatment is given or not is indicated by a binary variable D. A new observation is evaluated twice. Once with D=1 (treatment given) and once with D=0. Again the difference between the two predictions is the estimated uplift.


### What?
The last question we look at is "What marketing activity to choose?". There are many ways one could approach one customers (e.g. coupons, newsletters, ...). Finding the right method on an individual level is important, because different approaches might not only have different effects on potential customers but are also associated with different costs. For example it would be better to send a customer a newsletter which costs virtually nothing, rather then a coupon which would reduce the profit, if the newsletter has a similar effect on purchase probability. Since there isn't much research on this area and the selection of the proper marketing channel is crucial, we decided to lay the focus of our blog post on this issue.

# 3. Models <a class="anchor" id="models"></a>
## 3.1 Decision Trees Rzepakowski & Jaroszewicz <a class="anchor" id="decisiontree"></a>
### 3.1.1 Rzepakowski & Jaroszewicz Tree and Forest<a class="anchor" id="basic"></a>

In their paper <a  href="https://core.ac.uk/download/pdf/81899141.pdf/"> Decision trees for uplift modeling with single and multiple treatments</a> Rzepakowski and Jaroszewicz propose the usage of a decision tree for uplift modeling. The goal of their tree is to maximize the divergence of outcome distribution between the treatment(s) and control and between treatments.
To that end they developed a splitting criterion used to evaluate the possible splits. For each possible split they calculate the associated gain. To put it simply the gain is the divergence of outcome distribution after the split (conditional divergence), minus the diveregence prior to it (multiple divergence).
The aim is to find the split, which maximizes the gain. </br>
The formula for calculating the gain is given in Figure 6. $D$ represents a divergence function. In their paper they looked at KL-divergence, Euclidean distance and the chi-squared divergence. However, any other divergence measure could also be implemented. </br>
It is important to note here, that they only considere discrete outcome distributions in the paper. </br>
The gain:
\begin{equation}
D_{gain}(A) = D(P^{T_1}(Y),...,P^{T_k}(Y):P^C(Y)|A) - D(P^{T_1}(Y),...,P^{T_k}(Y):P^C(Y))
\end{equation}


Multiple Divergence:
\begin{equation}
{D(P^{T_1}(Y),...,P^{T_k}(Y):P^C(Y)|A) = \sum_a\frac{N(a)}{N}D(P^{T_1}(Y|a),...,P^{T_k}(Y|a):P^C(Y|a))}
\end{equation}

With $a$ being one outcome of a given test and $N(a)$ the number of samples with outcome $a$ of that test. Let's say we have $N = 1000$ people and we test for age below or above 25. Then $a_1$ would be $\leq25$ and $a_2$ would be $>25$. $N(a_1)$ are the number of people who are 25 or younger.

Conditional Divergence:

\begin{equation} 
D(P^{T\_1}(Y),...,P^{T\_k}(Y):P^{C}(Y))=\alpha\sum_{i=1}^{k}\lambda_iD(P^{T_i}(Y):P^{C}(Y))+(1-\alpha)\sum\_{i=1}^k\sum\_{j=1}^k\gamma\_{ij}D(P^{T_i}(Y):P^{T_j}(Y))
\end{equation}

There are 3 parameters which can be set by the user to adjust the model. </br>
$\alpha$: This parameter determines how much the treatment-control and between treatment divergence are measured. An $\alpha$ of 0.5 means both are valued equally. </br>

$\lambda_{i}$: Allows to put an emphasis on certain treatmens. For example one might put more emphasis on cheaper treatments. </br>

$\gamma_{ij}$: Allows to put individual weights on the divergence between treatments i and j. </br>

In addition to the gain, they also added a normalization factor. In the equation below it is given for KL-divergence and entropy. It is supposed to prevent bias towards test with high number of outcomes. Additionaly, it punishes uneven splits. </br>


\begin{equation}
I(A) =\alpha H(\frac{N^T}{N},\frac{N^C}{N})KL(P^T(A):P^C(A)) 
\end{equation}
\begin{equation}
+ (1 - \alpha)\sum_{i=1}^kH(\frac{N^{T_i}}{N^{T_i}+N^C},\frac{N^{C}}{N^{T_i}+N^C})KL(P^{T_i}(A):P^C(A))
\end{equation}
\begin{equation}
+\sum\_{i=1}^k\frac{N^{T_i}}{N}H(P^{T_i}(A))+\frac{N^{C}}{N}H(P^{C}(A))+\frac{1}{2}
\end{equation}


The first term measures the imbalance of the split between all the treatments combined and the control set. The second term measures the imbalance of the split for each treatment separately. The parameter $\alpha$ allows for setting the relative importance of those terms. The following two terms
penalize attributes with large numbers of values by summing the test entropy over all the
treatment and control datasets.</br>


Lastly, pruning based on a validation set is also implemented. The pruning algorithm goes through the entire tree starting at the bottom. For each subtree the algorithm checks if the outcome divergence in the leafs is greater than in the root for the validation set. If yes, than the algorithm continues, if no the subtree is pruned and the root becomes a new leaf. </br>

On the basis of this tree, we also implemented a function which allows to build a forest instead of just one tree. The implementation is loosely based on the random forest. There are two main parameters which can be set when building a forest. The number of trees and the number of covariates considered in each tree. For each tree a random subset of the covariates with the specified number of covariates is used.

```r
Forest building:

n_tree = 100 #The number of trees in the forest
n_features = 3 #The number of covariates used in each tree
forest = list()

Repeat #n_tree times:
  Randomly select #n_features of the covariates in the data and create a new subset of the data with those covariates
  Build a tree based on this subset
  Add the tree to the list of trees

For predictions:

predictions = list()

For each tree in forest:
  Make a prediction for the new sample and add it to the list

Return the average of all predictions

```

### 3.1.2 Simple Splitting Criterion <a class="anchor" id="simple"></a>

In addition to the tree proposed by Rzepakowski & Jaroszewicz we also implemented another splitting criterion as a benchmark. Unlike the previously discussed cruterion, ours aim to maximize the difference in mean outcome between treatment and control and between different treatments. This also allows for continuous outcomes without any adjustments.</br>
There is no pruning implemented as we wanted to keep it as simple as possible. For that reason we also only compare the difference in outcome of the left side of a new split to the root in order to get the gain of a give split. </br>
Despite our effort to keep this criterion simple we implemented a normalization factor which main use is to guarantee that we have at least one observation of each treatment and control in every leaf. In addition, it also punishes uneven splits.</br>
Here is the formula used to evaluate the possible splits. The "S = l" indicates, that we are only looking at the left side of the split.
$n\_{il}$ and $n\_{ir}$  are the number of samples with treatment i in the left and right leaf respectively. As one can see if either becomes 0 the whole equation is 0. 
\begin{equation}
\sum_{i=1}^{I}\sum\_{j=1}^{I}[Mean(Y|T_i=1,S=l)-Mean(Y|T_j=1,S=l)]^2*\prod\_{i=1}^I\frac{n\_{il}}{n}\*\frac{n\_{ir}}{n}
\end{equation}



## 3.2 Causal Tree and Causal Forest <a class="anchor" id="causaltree"></a>
The causal tree, introduced by Susan Athey et. al in their paper <a href = "https://github.com/susanathey/causalTree/blob/master/briefintro.pdf" target="_blank">An Introduction to Recursive Partitioning for Heterogeneous Causal Effect Estimation Using causalTree package</a> is a tree based classifier which directly estiamtes the treatment effect. It is based on the rpart package and implements many in the CART (Classification and Regression Trees). By default it only supports single treatment. Therefore, we train one tree for each of the multiple treatments and then compare the predicted uplifts. </br>
They also implemented a function which allows the user to build forests based on the causal tree. These forests are in the form of a list of rpart objects.

## 3.3 Separate Model <a class="anchor" id="separate"></a>

An early idea to model the uplift of a treated subject is known as the two-model approach. 
The name results from the fact that in the case of a single treatement assignment, a separate model is fitted for the treatment and control group.
<br />
By training a model only with treated subjects, it should incorporate the distinct features of the treatment in its outcomes (and vice versa when training it on control subjects).
Therefore, when predicting a subject with each model, the response of the models is expected to be its outcome when treated and when not treated. 
Using those two predictions the uplift can be calculated as the difference between the predicted outcome from the treatement and control model.
<br />
This approach extends naturally for the case of multiple treatments. For each treatment and the control group a separete response model is fitted. 
The expected uplift of a treatment, for a given individual is the difference between the outcome of the treatment's response model and the control model.

\begin{equation}
Uplift = P(Y^T | X) - P(Y^C | X)
\end{equation}

We adapt the naming convention from Zhao et al.(2017) and will refer to the two-model approach for multiple treatments as the separate model approach (SMA).
<br />
This approach can be used with any response model as a base learner. 
Therefore, its overall performance depends on the choice of the used model and the model specific parameter tuning.

<br />
Following we show our implementation the training of the SMA using Random Forests (RFs) as base learners. 
```r
# For each treatment and control in train data fit a model 
# and return list of trainedmodels
rf_models <- function(train_data, response, prediction_method){  
  treatments <- names(train_data)
  
  models <- list()
  for(i in c(1: length(treatments))){
    train <- train_data[[i]]
    
    # In case binary prediction, response needed as factor
    if(prediction_method == "class"){
      train[, response] <- as.factor(train[, response])
    }
    
    # Fit RF
    rf <- randomForest(as.formula(paste(response, "~.")), data = train, mtry=3, ntree = 300)
    
    models <- append(models, list(x= rf))
  }
  # Name list item with name of treatment / control
  names(models) <- treatments
  
  # Named list with fitted models for each treatment and control
  return(models)
}
```

Given the resulting list of trained models, the uplift can be calculated for a test dataset as follows.
```r
tree_sma_uplift <- function(models_dt, test_data, response, treatment, control_level, prediction_class){
  assignment <- test_data[ , treatment]
  outcome <- test_data[, response]
  
  test_data[ , treatment] <- NULL
  
  # Check whether regression or class prediction due to different dim of predict() result
  if(prediction_class == "class"){
    for (i in c(1 : length(models_dt))) {
      
      # Add all predictions to data frame
      if(i == 1){
        predictions <- data.frame(predict(models_dt[[i]], test_data, type = "prob")[ , 2])
      } else{
        predictions[ , paste0(i)] <- as.numeric(predict(models_dt[[i]], test_data, type = "prob")[ , 2] )
      }
    }
  # Regression prediction
  } else if(prediction_class == "anova"){
    for (i in c(1 : length(models_dt))) {
      
      # Add all predictions to data frame
      if(i == 1){
        predictions <- data.frame(predict(models_dt[[i]], test_data))
      } else{
        predictions[ , paste0(i)] <- as.numeric(predict(models_dt[[i]], test_data) )
      }
    }
  }
  
  # Rename columns to treatment names
  colnames(predictions) <- names(models_dt)
  
  k <- ncol(predictions)
  # For each treatment calculate the Uplift as T_i - Control
  for (i in c(1: k - 1)) {
    predictions[ , paste0("uplift_", names(models_dt)[i])] <- predictions[ , i] - predictions[ , k]
  }
  
  # Choose predicted treatment by the model
  predictions$T_index <- apply(predictions[, 1:3], 1, which.max)
  predictions$Treatment <- colnames(predictions)[predictions$T_index]
  
  # Add actual treatment assignment & outcome from test data
  predictions$Outcome <- outcome
  predictions$Assignment <- assignment
  predictions$Assignment <- ifelse(predictions$Assignment == control_level, "Control", as.character(predictions$Assignment))
  
  return(predictions)
}
```

<br />
The SMA represents an indirect uplift modelling approach, as the objective of the base learners is not to model the class difference, but the class specific outcomes.


<!--
#####
do they ???

Other related approaches which also extend naturally for multiple treatments are described in **Lo, Lai, but are not discussed in this work.
--> 


# 4 Evaluation Methods <a class="anchor" id="evaluationmethods"></a>

In contrast to a usual prediction task, it is not possible to observe the actual outome of a subject for all possible treatment assignments (including no treatment assignment).
Therefore, standard machine learning measures are not applicable to evaluate the performance of an uplift model.
In order to approximate the actual gains of using the predictions from an uplift model, some assumptions need to be made.
<br />
We describe two approaches from the literature which can be used to compare the performance of different uplift models. 

## 4.1 Uplift and Qini Curves<a class="anchor" id="uplift_curves"></a>

<!--
Shorten only for Uplift Curves ???
-->

Both uplift curves and qini curves build upon the idea of cumulative gain charts.
Those require all targets to be ranked in descending order by their score. 
For the single treatment case the score represents the uplift when assigning the treatment to the subject.
In the case of multiple treatments the uplift model predicts the uplift for each treatment. The score then corresponds to the maximal uplift of all possible treatment assignments.
In case the uplift for all treatments is negative, the score can also be negative. This corresponds to the behavior of the 'Do-Not-Disturbs' group.
<br />
Given the ranked outcomes for each treatment and the control group, a cumulative lift curve can be drawn. 
The x axis represents the number of treated subjects and the y axis the cumulative outcome.
The uplift curve represents the lift curve for a treatment, subtracted by the lift curve of the control group.
The main assumption of this approach is that similarly scored groups inhibit similar features and are therefore comparable.
However, Rzepakowski et al. (2012) mention that this property is not necessarly satisfied, but currently no better alternatives exist.
<br />
In the case of multiple possible treatment assignments, naturally for each treatment a separate uplift curve can be drawn. 
In order to combine those curves it is possible to use the maximal uplift curve for each decile, under the assumption that this corresponds to always assigning the highest predicted treatment to a subject.
<br />
As the treatments and control groups in the testing dataset can have different sizes, their outcomes need to be scaled in order to represent a meaningfull subtration of the treatment and control outcomes.
Here the difference between the qini and uplift curves becomes apparent.
The qini curve as described by Radcliffe et al. (2007) scales the outcomes to the sample sizes of the treatment group.


\begin{equation}
f(t) = Y_t^T - \frac{Y_t^C N_t^T}{N_t^C}
\end{equation}

With $t$ as the quantile for the amount of the treated population, $T$ as the treatment assigned and $C$ as the Control group.
$N_t^T$ and $N_t^C$ depict the number of subjects in the according treatment anf contol group, within the top quantile. 
$Y_t^T$ and $Y_t^C$ stand for the total response in the treated group within the top quantile.
This formulation 

<br />
The uplift curve, as described in Gutierrez et al. (2017), uses the sample mean of each group and scales it to the full sample size. 

\begin{equation}
g(t) = ( \frac{Y_t^T }{N_t^T} - \frac{Y_t^C }{N_t^C} ) (N_t^T + N_t^C)
\end{equation}

Both the Qini and Uplift curve return a separate curve for each treatment.
For multiple treatments we have adjusted the uplift curve as follows:

\begin{equation}
g(t) = ( \frac{Y_t^{T^\prime} }{N_t^{T^\prime}} - \frac{Y_t^C }{N_t^C} ) (N_t)
\end{equation}

We denote the combined group of all possible treatments with $T^\prime$. This group is than scored and ranked as a whole by their maximal uplift value. 
Additionaly, the uplift per subject is than upscaled for the whole testing group $N_t$.

<br />
As the qini curves are scaling towards the treatment sample size, a combination of multiple treatment groups with different sizes becomes more cumbersome. 
Therefore, we see the uplift curves more applicable for multiple treatments uplift evaluation and will omit the qini curves in our experimental evaluations.


<!--
 plotting combined treatment curve...
 --> 

 
<!--

DOWNSIDES OF THOSE APPROACHES ???

 --> 


 
## 4.2 Expected Outcome <a class="anchor" id="expected_outcome"></a>

Method from (Zhao et al. 2017)


\begin{equation}
Z = \sum_{t=0}^{K}\frac{1}{p_t}Y I \{ h(X)=t \} I \{ T=t \}
\end{equation}

\begin{equation}
E[Z] = E[Y|T=h(X)]
\end{equation}

\begin{equation}
\bar{z}=\frac{1}{N} \sum_{i=1}^{N}z^{(i)} 
\end{equation}

The expected response can also be used to draw uplift curves. In Zhao et al. (2017) those curves are named <i>modified uplift curves</i>.
Once again the subjects are ranked by their score (expected uplift). Now for the top x-percent the optimal treatment is assigned and the remaining subject are assigned to the control group.
Now the expected value for treating x-percent of the subjects are given as $\bar{z}$.
<br />
In contrast to the qini and uplift curves, the modified uplift curve is not cumulative, as the outcomes of the not treated subjects are also taken into account. 
Also the expected outcome is defined per customer and therefore indipendent from the size of the testing data.




# 5. Experimental Setup <a class="anchor" id="outlook"></a>
## 5.1 Data Sets <a class="anchor" id="datasets"></a>

<!--
Describe Dataset with different outcome variables
-->

<a href="https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html" target="_blank">Kevin Hillstrom’s MineThatData</a>


# 6. Results <a class="anchor" id="evaluationresults"></a>

Models evaluated using 5-fold cross validation.


## Predictive Results


## Training Duration
Even though predictive performance is the focus of this evaluation we also wanted to look at how well our approaches scaled in terms of training time. The are 3 factors we looked at which will influence performance time: number of observations, number of covariates and number of treatments.</br>
The figures below show a comparisson of our models.</br>
3 Covariates: </br>
<img
align="center"
width="550"
height="360"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Comparison3Features.png"> </br>
4 Covariates: </br>
<img
align="center"
width="550"
height="360"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Comparison4Features.png"> </br>
5 Covariates: </br>
<img
align="center"
width="550"
height="360"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Comparison5Features.png"></br>


# 7. Outlook <a class="anchor" id="outlook"></a>

The described evaluation metrics for uplift models all represent approximative approaches under specific assumptions. 
The expected response metric from Zhao et al. (2017) is the only metric defined under the assumptions of multiple treatements.
A standarized evaluation procedure in the uplift field could increase the comparability of models throughout different works.


# 8. References <a class="anchor" id="references"></a>

* Devriendt, F., Moldovan, D. and Verbeke, W., 2018. A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: A stepping stone toward the development of prescriptive analytics. Big data, 6(1), pp.13-41.
* Gubela, R., Bequé, A., Lessmann, S. and Gebert, F., 2019. Conversion uplift in e-commerce: A systematic benchmark of modeling strategies. International Journal of Information Technology & Decision Making (IJITDM), 18(03), pp.747-791.
* Gutierrez, P. and Gérardy, J.Y., 2017, July. Causal inference and uplift modelling: A review of the literature. In International Conference on Predictive Applications and APIs (pp. 1-13).
* Lai, Y.T., Wang, K., Ling, D., Shi, H. and Zhang, J., 2006, December. Direct marketing when there are voluntary buyers. In Sixth International Conference on Data Mining (ICDM'06) (pp. 922-927). IEEE.
* Lo, V.S. and Pachamanova, D.A., 2015. From predictive uplift modeling to prescriptive uplift analytics: A practical approach to treatment optimization while accounting for estimation risk. Journal of Marketing Analytics, 3(2), pp.79-95.
* Radcliffe, N.J., 2007. Using control groups to target on predicted lift: Building and assessing uplift models. Direct Marketing Analytics Journal, 1, p.1421.
* Rzepakowski, P. and Jaroszewicz, S., 2012. Decision trees for uplift modeling with single and multiple treatments. Knowledge and Information Systems, 32(2), pp.303-327.
* Zhao, Y., Fang, X. and Simchi-Levi, D., 2017, June. Uplift modeling with multiple treatments and general response types. In Proceedings of the 2017 SIAM International Conference on Data Mining (pp. 588-596). Society for Industrial and Applied Mathematics.
* Dekimpe, Marnik & Franses, Ph.H.B.F. & Hanssens, Dominique & Naik, Prasad. (2006). Time-Series Models in Marketing. 