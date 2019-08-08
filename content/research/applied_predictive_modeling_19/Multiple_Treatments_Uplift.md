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

1. [Introduction](#introduction)
2. [Common Marketing Challenges](#challenges)
3. [Motivation for Multiple Treatments](#motivation)
4. [Models](#models) </br>
4.1 [Decision Trees Rzepakowski & Jaroszewicz](#decisiontree) </br> 
4.1.1 [Basic Rzepakowski & Jaroszewicz](#basic) </br>
4.1.2 [Simple Splitting Criterion](#simple) </br>
4.2 [Causal Tree and Causal Forest](#causaltree) </br>
4.3 [Separate Model](#separate) </br>
5. [Evaluation](#evaluation)</br>
5.1 [Methods](#evaluationmethods)</br>
5.2 [Results](#evaluationresults)
6. [Outlook](#outlook)
7. [References](#references)

# 1. Introduction <a class="anchor" id="introduction"></a>
Nowadays marketing practitioners face a multitude of challenges. 

In the first part of this analysis we will look at which challenges exist and how data analysis might be employed to tackle these problems.

In the second part we will look more closely on one specific problem and give a summary and an evaluation of proposed methods to solve that problem.
# 2. Common Marketing Challenges <a class="anchor" id="challenges"></a>
Some of the most commonly cited challenges we found were:
- Drive Traffic to the website
- Increase/Prove ROI of marketing campaigns
- Turn website visitors into buyers
- Reach people at the right time
- Target content to the right audience
- Select marketing channel/method

Since those challenges were all related to marketing activity we decided to focus on how data anlysis could be used to improve the performance of marketing activities. More specificly there are 3 questions with regards to marketing activity we focused on.
### When to start the marketing activity?
When it comes to selecting the time of a marketing activity there is not much research out there with regards to treatment effects.
Most advice for marketing pratitioners focuses on social media marketing and on what time are best for publishing new posts. These guideline mostly look at single performance measures like engagement to see at which times the performance measure are maximized. 
<a  href="https://sproutsocial.com/insights/best-times-to-post-on-social-media/"> Best times to post on social media for 2019</a>. </br>
In their paper <a href = "https://www.researchgate.net/publication/4753376_Time-Series_Models_in_Marketing"> Time Series Models in Marketing </a> the authors look at the application of time series models for marketing. For example they use persistance modeling in order to estimate  the longterm effect of marketing activities.

<img
align="center"
width="300"
height="200"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/TimeSeries.PNG">

Being able to estimate the long term effect of ones marketing activity allows the practicioner to select the appropriate starting point in order to maximize the return.
### Who to target?
The importance of this question varies greatly depending on the kind of marketing that is being done. Figure 2 shows various types of marketing from broad to narrow. The narrower the more potential there is for the usage of treatment effects. For the broadest possible marketing activity the average treatment effect (ATE) is important but no selection cade be made in terms of who we target. Narrower activities might allow us to select certain subgroups of our potential customers. Here we would be interested in the group average treatment effects (GATES) of those subgroups. Then we could determine which group to target based on those treatment effects.

<img
align="center"
width="300"
height="200"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Market-Targeting-Strategies.png">

Our focus lies on the most narrow kind of marketing activities. Here we can decide on an individual basis whether we target a given potential customer. Historically practicioners would target the people who they thought would be most likely to do a purchase. This approach is suboptimal since it is solely based on the overall purchase possibility and not the effect of the treatment. </br>
In general we can separate our customers in 4 groups (Figure 3). </br>

<img
align="center"
width="300"
height="200"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Mike-Thurber-Graphic-2.png">

With the historical approach we will target mostly the 'Sure Things' and maybe the 'Do-Not-Disturbs'. For those groups we at best get no return and at worst actually lose customers. Ideally we want to target the 'Persuadables'. This is commonly done by estimating the conditional average treatment effect (CATE) or uplift of the proposed marketing activity and the target the customers where the activity is estimated to have the highest effect. </br>
Several approaches have been proposed to estimate uplift. Gubela et. al give an overview in their paper <a href = "https://www.researchgate.net/publication/331791032_Conversion_uplift_in_E-commerce_A_systematic_benchmark_of_modeling_strategies"> Conversion uplift in E-commerce: A systematic benchmark of modeling strategies</a>. In their evaluation they find that the two model uplift method and interaction term method (ITM) performed best. </br>
The two model approach as the name suggests uses two separate models. The training set is split into two with one set contained all the treated observations and the other all control observations. Then for each traning set one model is built to predict the outcome. To estimate the uplift of the treatment for a new person, we generate the predicted outcome with both models. One predicted outcome with treatment and one without. The difference between these two estimates is the expected uplift. The two model is very simple and works with virtually every possible base model (e.g. random forest, linear regression, svm, ...). Since it is so simple it is often considered a good benchmark to test new approaches against.</br>
The interaction term method (ITM) was proposed by Lo in his 2002 paper <a href="https://www.researchgate.net/publication/220520042_The_True_Lift_Model_-_A_Novel_Data_Mining_Approach_to_Response_Modeling_in_Database_Marketing">The True Lift Model - A Novel Data Mining Approach to Response Modeling in Database Marketing</a>. Unlike double machine learning, ITM only uses a single model. However, ITM also works with two predictions. One with treatment and one without. Both predictions are obtained from the same model which has been trained on both treatment and control data. Whether a treatment is given or not is indicated by a binary variable D. A new observation is evaluated twice. Once with D=1 (treatment given) and once with D=0. Again the difference between the two predictions is the estimated uplift.


### Which treatment should be used for which target?


# 3. Motivation for Multiple Treatments <a class="anchor" id="motivation"></a>
We decided to focus on multiple treatments for several reasons. Today there are many ways to reach your potential customers and picking the right one is crucial for succesfull marketing activity. Additionally, we found that there has not been much research on this topic and there is no comprehensive comparison of the research that does exist. Therefore, we decided to look at the models that have been proposed so far and compare them. Furthermore, we want to identify potential new directions for further research as it was done in the blog post 

  
# 4. Models <a class="anchor" id="models"></a>
## 4.1 Decision Trees Rzepakowski & Jaroszewicz <a class="anchor" id="decisiontree"></a>
### 4.1.1 Rzepakowski & Jaroszewicz Tree and Forest<a class="anchor" id="basic"></a>

In their paper <a  href="https://core.ac.uk/download/pdf/81899141.pdf/"> Decision trees for uplift modeling with single and multiple treatments</a> Rzepakowski and Jaroszewicz propose the usage of a decision tree for uplift modeling. The goal of their tree is to maximize the divergence of outcome distribution between the treatment(s) and control and between treatments.
To that end they developed a splitting criterion used to evaluate the possible splits. For each possible split they calculate the associated gain. To put it simply the gain is the divergence of outcome distribution after the split, minus the diveregence prior to it.
The aim is to find the split, which maximizes the gain. </br>
The formula for calculating the gain is given in Figure 6. The 'D' represents a divergence function. In their paper they looked at KL-divergence, Euclidean distance and the chi-squared divergence. However, any other divergence measure could also be implemented. </br>
It is important to note here, that they only considere discrete outcome distributions in the paper. </br>
The gain:
<img
align="center"
width="600"
height="100"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Gain.PNG">

Multiple Divergence:
<img
align="center"
width="600"
height="100"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Multiple.PNG">

Conditional Divergence:
<img
align="center"
width="600"
height="100"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/Conditional.PNG">

As one can see there are 3 parameters which can be set by the user to adjust the model. </br>
$\alpha$: This parameter determines how much the treatment-control and between treatment divergence are measured. An $\alpha$ of 0.5 means both are valued equally. </br>
$\lambda$: Allows to put an emphasis on certain treatmens. For example one might put more emphasis on cheaper treatments. </br>
$\gamma$: Allows to further emphasize selected between treatment divergences. </br>

In addition to the gain, they also added a normalization factor which has two functions. Firstly, it is supposed to prevent bias towards test with high number of outcomes. Secondly, it punishes uneven splits. </br>

Lastly, pruning based on a validation set is also implemented. The pruning algorithm goes through the entire tree starting at the bottom. For each subtree the algorithm checks of the outcome divergence in the leafs is greater than in the root for the validation set. If yes, than the algorithm continues, if no the subtree is pruned and the root becomes a new leaf. </br>

On the basis of this tree, we also implemented a function which allows to build a forest instead of just one tree. The implementation is loosely based on the random forest. There are two main parameters which can be set when building a tree. The number of trees and the number of covariates considered in each tree. For each tree a random subset of the covariates with the specified number of covariates is used.

### 4.1.2 Simple Splitting Criterion <a class="anchor" id="simple"></a>

In addition to the tree proposed by Rzepakowski & Jaroszewicz we also implemented another splitting criterion as a benchmark. Unlike the previously discussed cruterion, ours aim to maximize the difference in mean outcome between treatment and control and between different treatments. This also allows for continuous outcomes without any adjustments.</br>
There is no pruning implemented as we wanted to keep it as simple as possible. For that reason we also only compare the difference in outcome of the left side of a new split to the root in order to get the gain of a give split. </br>
Despite our effort to keep this criterion simple we implemented a normalization factor which main use is to guarantee that we have at least one observation of each treatment and control in every leaft. In addition, it also punishes uneven splits.

<img
align="center"
width="700"
height="100"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/SimpleCriterionCategorical.PNG">


## 4.2 Causal Tree and Causal Forest <a class="anchor" id="causaltree"></a>
The causal tree, introduced by Susan Athey et. al in their paper <a href = "https://github.com/susanathey/causalTree/blob/master/briefintro.pdf">An Introduction to Recursive Partitioning for Heterogeneous Causal Effect Estimation Using causalTree package</a> is a tree based classifier which directly estiamtes the treatment effect. It is based on the rpart package and implements many in the CART (Classification and Regression Trees). By default it only supports single treatment. Therefore, we train one tree for each of the multiple treatments and then compare the predicted uplifts. </br>
They also implemented a function which allows the user to build forests based on the causal tree. These forests are in the form of a list of rpart objects.

## 4.3 Separate Model <a class="anchor" id="separate"></a>
# 5. Evaluation <a class="anchor" id="evaluation"></a>
## 5.1 Methods <a class="anchor" id="evaluationmethods"></a>
## 5.2 Results <a class="anchor" id="evaluationresults"></a>
### Predictive Results
### Training Duration
Even though predictive performance is the focus of this evaluation we also wanted to look at how well our approaches scaled in terms of training time. The are 3 factors we looked at which will influence performance time: number of observations, number of covariates and number of treatments.</br>
The figures below show a comparisson of our methods.</br>
<img
align="center"
width="225"
height="150"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/2Treatments.png">
<img
align="center"
width="225"
height="150"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/3Treatments.png">
<img
align="center"
width="225"
height="150"
style="display:block;margin:0 auto;" src="/blog/img/seminar/multiple_treatment_uplift/4Treatments.png">
# 6. Outlook <a class="anchor" id="outlook"></a>



# 7. References <a class="anchor" id="references"></a>

* Devriendt, F., Moldovan, D. and Verbeke, W., 2018. A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: A stepping stone toward the development of prescriptive analytics. Big data, 6(1), pp.13-41.
* Gubela, R., Bequé, A., Lessmann, S. and Gebert, F., 2019. Conversion uplift in e-commerce: A systematic benchmark of modeling strategies. International Journal of Information Technology & Decision Making (IJITDM), 18(03), pp.747-791.
* Lai, Y.T., Wang, K., Ling, D., Shi, H. and Zhang, J., 2006, December. Direct marketing when there are voluntary buyers. In Sixth International Conference on Data Mining (ICDM'06) (pp. 922-927). IEEE.
* Lo, V.S. and Pachamanova, D.A., 2015. From predictive uplift modeling to prescriptive uplift analytics: A practical approach to treatment optimization while accounting for estimation risk. Journal of Marketing Analytics, 3(2), pp.79-95.
* Rzepakowski, P. and Jaroszewicz, S., 2012. Decision trees for uplift modeling with single and multiple treatments. Knowledge and Information Systems, 32(2), pp.303-327.
* Zhao, Y., Fang, X. and Simchi-Levi, D., 2017, June. Uplift modeling with multiple treatments and general response types. In Proceedings of the 2017 SIAM International Conference on Data Mining (pp. 588-596). Society for Industrial and Applied Mathematics.
