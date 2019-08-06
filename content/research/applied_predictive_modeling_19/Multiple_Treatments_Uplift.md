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
4.2 [Causal Tree](#causaltree) </br>
4.3 [Causal Forest](#causalforest) </br>
4.4 [Separate Model](#separate) </br>
5. [Evaluation](#evaluation)
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
style="display:block;margin:0 auto;" src="https://github.com/jkrol21/blog/blob/master/static/img/seminar/multiple_treatment_uplift/TimeSeries.PNG">

Being able to estimate the long term effect of ones marketing activity allows the practicioner to select the appropriate starting point in order to maximize the return.
### Who to target?
The importance of this question varies greatly depending on the kind of marketing that is being done. Figure 2 shows various types of marketing from broad to narrow. The narrower the more potential there is for the usage of treatment effects. For the broadest possible marketing activity the average treatment effect (ATE) is important but no selection cade be made in terms of who we target. Narrower activities might allow us to select certain subgroups of our potential customers. Here we would be interested in the group average treatment effects (GATES) of those subgroups. Then we could determine which group to target based on those treatment effects.

<img
align="center"
width="300"
height="200"
style="display:block;margin:0 auto;" src="https://github.com/jkrol21/blog/blob/master/static/img/seminar/multiple_treatment_uplift/Market-Targeting-Strategies.png">

Our focus lies on the most narrow kind of marketing activities. Here we can decide on an individual basis whether we target a given potential customer. Historically practicioners would target the people who they thought would be most likely to do a purchase. This approach is suboptimal since it is solely based on the overall purchase possibility and not the effect of the treatment. </br>
In general we can separate our customers in 4 groups (Figure 3). </br>

<img
align="center"
width="300"
height="200"
style="display:block;margin:0 auto;" src="https://github.com/jkrol21/blog/blob/master/static/img/seminar/multiple_treatment_uplift/Mike-Thurber-Graphic-2.png">

With the historical approach we will target mostly the 'Sure Things' and maybe the 'Do-Not-Disturbs'. For those groups we at best get no return and at worst actually lose customers. Ideally we want to target the 'Persuadables'. This is commonly done by estimating the conditional average treatment effect (CATE) or uplift of the proposed marketing activity and the target the customers where the activity is estimated to have the highest effect. </br>
Several approaches have been proposed to estimate uplift. Gubela et. al give an overview in their paper <a href = "https://www.researchgate.net/publication/331791032_Conversion_uplift_in_E-commerce_A_systematic_benchmark_of_modeling_strategies"> Conversion uplift in E-commerce: A systematic benchmark of modeling strategies</a>. In their evaluation they find that the two model uplift method and interaction term method (ITM) performed best. </br>



### Which treatment should be used for which target?


# 3. Motivation for Multiple Treatments <a class="anchor" id="motivation"></a>
We decided to focus on multiple treatments for several reasons. Today there are many ways to reach your potential customers and picking the right one is crucial for succesfull marketing activity. Additionally, we found that there has not been much research on this topic and there is no comprehensive comparison of the research that does exist. Therefore, we decided to look at the models that have been proposed so far and compare them. Furthermore, we want to identify potential new directions for further research as it was done in the blog post 

  
# 4. Models <a class="anchor" id="models"></a>
## 4.1 Decision Trees Rzepakowski & Jaroszewicz <a class="anchor" id="decisiontree"></a>
### 4.1.1 Rzepakowski & Jaroszewicz Tree and Forest<a class="anchor" id="basic"></a>

In their paper <a  href="https://core.ac.uk/download/pdf/81899141.pdf/"> Decision trees for uplift modeling with single and multiple treatments</a> Rzepakowski and Jaroszewicz propose the usage of a decision tree for uplift modeling. The goal of their tree is to maximize the divergence of outcome distribution between the treatment(s) and control and between treatments.
To that end they developed a splitting criterion used to evaluate the possible splits. For each possible split they calculate the associated gain (Figure 5)

<img
align="center"
width="600"
height="100"
style="display:block;margin:0 auto;" src="https://github.com/jkrol21/blog/blob/master/static/img/seminar/multiple_treatment_uplift/Gain.PNG">

<img
align="center"
width="600"
height="100"
style="display:block;margin:0 auto;" src="https://github.com/jkrol21/blog/blob/master/static/img/seminar/multiple_treatment_uplift/Multiple.PNG">

<img
align="center"
width="600"
height="100"
style="display:block;margin:0 auto;" src="https://github.com/jkrol21/blog/blob/master/static/img/seminar/multiple_treatment_uplift/Conditional.PNG">

### 4.1.2 Simple Splitting Criterion <a class="anchor" id="simple"></a>

<img
align="center"
width="600"
height="100"
style="display:block;margin:0 auto;" src="https://github.com/jkrol21/blog/blob/master/static/img/seminar/multiple_treatment_uplift/SimpleCriterionCategorical.PNG">


## 4.2 Causal Tree <a class="anchor" id="causaltree"></a>
## 4.3 Causal Forest <a class="anchor" id="causalforest"></a>
## 4.4 Separate Model <a class="anchor" id="separate"></a>
# 5. Evaluation <a class="anchor" id="evaluation"></a>
# 6. Outlook <a class="anchor" id="outlook"></a>

![](/blog/img/seminar/multiple_treatment_uplift/result1.png)

# 7. References <a class="anchor" id="references"></a>
