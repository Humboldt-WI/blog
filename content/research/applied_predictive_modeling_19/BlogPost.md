+++
title = "APA Template"
date = '2019-08-15'
tags = [ "Causal Inference", "Class19",]
categories = ["Course projects"]
banner = "img/seminar/nn_fundamentals/neuralNetworkKlein.jpg"
author = "Seminar Applied Predictive Modeling (SS19)"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = "One sentence description of your work"
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
4.1.2 [Rzepakowski & Jaroszewicz Forest](#forest) </br> 
4.1.3 [Simple Splitting Criterion](#simple) </br>
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
<a  href="https://sproutsocial.com/insights/best-times-to-post-on-social-media/"> Best times to post on social media for 2019 </a>. </br>
In their paper <a href = "https://www.researchgate.net/publication/4753376_Time-Series_Models_in_Marketing"> Time Series Models in Marketing </a> The authors look at the application of time series models for marketing. For example they use persistance modeling in order to estimate  the longterm effect of marketing activities.
<img
align="center"
width="300"
height="360"
style="display:block;margin:0 auto;" src="https://github.com/jkrol21/blog/blob/master/docs/img/seminar/BilderAPABlog/TimeSeries.PNG"> </br> 
Figure 1: Estimation of long term effects</img> </br>
Being able to estimate the long term effect of ones marketing activity allows the practicioner to select the appropriate starting point in order to maximize the return.
### Who to target?
The importance of this question varies greatly depending on the kind of marketing that is being done. 
### Which treatment should be used for which target?


# 3. Motivation for Multiple Treatments <a class="anchor" id="motivation"></a>
We decided to focus on multiple treatments for several reasons. Today there are many ways to reach your potential customers and picking the right one is crucial for succesfull marketing activity. Additionally, we found that there has not been much research on this topic and there is no comprehensive comparison of the research that does exist. Therefore, we decided to look at the models that have been proposed so far and compare them. Furthermore, we want to identify potential new directions for futher research as it was done in the blog post 

  
# 4. Models <a class="anchor" id="models"></a>
## 4.1 Decision Trees Rzepakowski & Jaroszewicz <a class="anchor" id="decisiontree"></a>
### 4.1.1 Basic Rzepakowski & Jaroszewicz <a class="anchor" id="basic"></a>
### 4.1.2 Rzepakowski & Jaroszewicz Forest <a class="anchor" id="forest"></a>
### 4.1.3 Simple Splitting Criterion <a class="anchor" id="simple"></a>
## 4.2 Causal Tree <a class="anchor" id="causaltree"></a>
## 4.3 Causal Forest <a class="anchor" id="causalforest"></a>
## 4.4 Separate Model <a class="anchor" id="separate"></a>
# 5. Evaluation <a class="anchor" id="evaluation"></a>
# 6. Outlook <a class="anchor" id="outlook"></a>
# 6. References <a class="anchor" id="references"></a>
