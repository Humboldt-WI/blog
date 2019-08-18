+++
title = "Causal KNN"
date = '2019-08-01'
tags = [ "Causal Inference", "Class19", "Causal Inference", "Uplift Modelling", "KNN Algorithm",]
categories = ["Course projects"]
banner = "img/seminar/sample/hu-logo.jpg"
author = "Maximilian Kricke, Tim Peschenz"
disqusShortname = "https-wisample-github-io-blog"
description = "Blog Post for Seminar Applied Predictive Analytics"
+++

<head>
<script type="text/javascript" src="http://latex.codecogs.com/latexit.js"></script>
<script type="text/javascript">
LatexIT.add('p',true);
</script>
</head>



# Applied Predictive Analytics Seminar - Causal KNN 



Beyond estimating the overall effect of a treatment, the uplift, econometric and statistical literature have set their eyes on estimating the personal treatment effect for each individual. This blogpost highly relates to the paper of Hitsch & Misra (2018), where a novel, direct uplift modeling approach is introduced, called Causal KNN. The k-nearest neighbour algorithm provides an interesting opportunity for the estimation of treatment effects in small groups. To estimate the treatment effect, the Transformed Outcome, introduced by Athey and Imbens (2015b), is used. We will show that the Transformed Outcome is not only restricted to the Causal KNN modeling approach. Under several assumptions to the data, the Transformed Outcome equals in expectation to the true CATE. Hence it can be seen as a general tool for the parameter tuning in the context of Treatment Effect Analysis. Additionally, the Transformed Outcome can be used to evaluate different uplift models in terms of the MSE. 

In the beginning, we will give a motivation for the topic, followed by an introduction of the methods presented by Hitsch & Misra (2018). In order to highlight the applicability of those methods, an implementation on real world data of a randomized E-Mail marketing campaign is given. We will perform the Causal KNN estimation as well as a Causal Tree estimation. In the end, the model performance is compared using MSE, AUUC and the Qini-Curve.  

## Motivation and Literature Foundation:

The analysis of causal effects is of high importance in many different areas. The testing of a new drug, the evaluation of a labor market program, or the optimization of a marketing campaign in a business setting are all influenced by an unobservable bias. Researchers in all these areas are facing the fundamental problem of causal inference, which is the fact that we never observe the individual treatment effect, but the outcome corresponding only to the assigned treatment. To grasp the concept of the Causal KNN the introduction of the potential outcome framework first given by Rubin (1974, 1977, 1978) is required: For every individual $i$ with the attributes $X_i$ in the population sample, we observe the outcome $Y_i(W_i)$ that corresponds to a specific treatment status $W_i=\text{{0,1}}$. For the given sample, the individual probability of receiving a treatment conditional on the individuals attributes is called the propensity score $P(W_i=1|X_i=x)=\mathbb{E}[W_i=1|X_i=x]=e(x)$ (Rosenbaum and Rubin, 1983). In this general framework, causal effects can be understood as comparisons of potential outcomes defined on the same unit.

This blogpost highly relates to the work "Heterogeneous Treatment Effect Estimation" of Hitsch & Misra (2018). Their work has a broad framework and deals with a variety of topics in the Treatment Effect Estimation especially in the business setting. 
In the first part, they are presenting an approach to compare several different targeting policies by using only one single randomized sample. They propose an inverse probability weighted estimator which is unbiased, qualitatively equivalent to a field experiment and cost effective. Maximizing the profit of a marketing campaign comes down to the knowledge of the Conditional Average Treatment Effect (CATE) $${\tau(x)}={\tau(x)}^{CATE}=\mathbb{E}[Y_i(1)-Y_i(0) | X_i=x]$$ 
As it follows from the fundamental problem of causal inference, the true CATE is not observable in general. Therefore the main part of their work is devoted to identification methods for the true CATE. They introduce three crucial assumptions to the data, to make the CATE identifiable.

**Unconfoundedness:**

$Y_i(0), Y_i(1) \bot W_i | X_i$. 

It means, that ones individuals outcome does not depend on the treatment conditional on the covariates (Rubin, 1990). 'The treatment assignment is random within a subset of customers with identical features' (Hitsch & Misra, 2018). This condition can be fulfilled by using a randomized sample set. In marketing, it is often possible to address campaigns randomized, therefore this condition should be satisfied in general and especially in our application. We will see that this assumption is crucial for the derivation of an unbiased estimator for the CATE. 

**The Overlap Assumption:**

$0 < e(x) < 1$

For general application, the propensity score has to be defined strictly between zero and one. If this assumption is violated, there would be no treated or untreated observation for specific values of the feature vector $X_i$. We will see that this assumption is also required to derive an estimator for the CATE. 

**Stable Unit Treatment Value Assumption (SUTVA):**

The treatment received by customer i, $W_i$, has no impact on the behaviour of any other customer. There are no social interaction or equilibrium effects among the observed units. According to Hitsch and Misra, the SUTVA assumption should be satisfied in general marketing applications. It appears implausible to observe economically significant social effects, e.g. word-of-mouth,  due to the receipt of a newsletter.   

If those three assumptions are satisfied, we can calculate the true CATE by taking the mean difference between the outcome of treated and untreated units with identical features:

$\tau(x) = \mathbb{E}[Y_i|X_i,W_i = 1] - \mathbb{E}[Y_i|X_i,W_i = 0]$

It is unrealistic to assume that there are more than one customer with exactly the same feature set, especially for high dimensional data. And therefore the true CATE as given in the above formula needs to be estimated. Hitsch and Misra are presenting methods to predict the CATE indirectly via the outcome variable, i.e. $\mathbb{E}[(Y_i-\hat{\mu}(X_i,W_i))^2]$ (OLS, Lasso, Logit, Non-parametric models) as well as directly via the treatment effect, i.e. $\mathbb{E}[(\tau(X_i)-\hat{\tau}(X_i))^2]$ (Causal KNN, Treatment Effect Projection, Causal Forest). The aim of this blogpost is to explain and apply the Causal KNN estimation method that directly predict the individual incremental effect of targeting.  Hitsch and Misra (2018) describe the indirect approaches as conceptually wrong in the field of uplift estimation. They recommend the usage of direct estimation methods, since they yield a more accurate estimation of the treatment effect.  

To illustrate the proposed Causal KNN method to estimate the causal effects, we will utilize a business framework. Imagine: A company runs a marketing campaign. 
Now this campaign can consist of various customer treatments, such as a targeted price, a coupon or a newsletter. But how do you evaluate the campaigns' success? 
At this point, Hitsch & Misra (2018) elaborate the dependency of marketing success from the individual targeting effort cost. A customer should be targeted if the individual profit contribution is greater than the targeting cost for this individual. It is shown that the CATE on an individual basis is sufficient to construct an optimal
targeting policy. To then conduct a profit evaluation, the individual costs and margins are required, which are generally known for a company, but unknown in our data. Therefore, we will focus on estimating the causal effects without including the individual targeting costs by using Causal KNN in combination with the transformed outcome approach. In the business framework, the expected causal effects can be described through the CATE, which represents the incremental targeting effort effect.

<!-- ## Summary of the Literature Foundation
<!-- This blogpost has strong relations to the work of Hitsch & Misra (2018). 
Causal Inference is done in the potential outcome framework. This framework will be briefly explained in the following. Assume a company is about to run an E-Mail campaign. Therefore, the company needs to decide on the amount of people that will receive an E-Mail. This decision depends on the cost of the campaign, compared to the expected revenue. On a customer level, this decision boils down from the comparison of the targeting effort cost, to the incremental profit contribution of each customer. The fundamental problem of treatment effect analysis is, that it can not be observed, whether a customers action was influenced by the received treatment or not. It can not be verified how a treated customer would have acted, when he or she would not have received such an E-Mail. Therefore we aim at identifying the underlying factors of the customers behaviour, to eliminate these effects for the calculation of the treatment effect estimation. A given targeting policy cannot be directly evaluated in a randomized sample, because for many customers the proposed treatment assignment by the targeting policy will differ from the realized treatment assignment. However, for units where both, the proposed and realized treatment assignment agree, we can rescale the realized profit contribution by the inverse of the propensity score, to account for the percentage of similar units that are "missing", in the sense that the proposed and realized treatment assignment are different. This problem can be described via the following graphic: -->

<!-- 
<img align="center" width="800"
style="display:block;margin:0 auto;" 
src="/blog/img/seminar/causal_knn/treatment_matrix.PNG"
alt = "Classification of customer base along two dimensions: Targeting and Reaction"> -->


 

<!-- To evaluate the treatment effect estimations, there are multiple approaches in the marketing literature (see e.g. Gutierrez and Gérardy, 2016). An observed uplift data set has to consists of the observations $D = (Y_i, X_i, W_i)^N$, where $N$ is the number of customers, $Y_i$ is the target variable, $W_i$ the binary treatment variable (0,1) and $X_i$ represents the covariates vor each unit. 

* Average Treatment effect: $E(Y_i(1)-Y_i(0)$
* Average Treatment effect on the treated: $E(Y_i(1)-Y_i(0) | W_i = 1)$
* Conditional Average Treatment effect: $E(Y_i(1)-Y_i(0) | X_i=x)$

where $Y_i(1)$ and $Y_i(0)$ is the observed outcome for the treated ($W_i=1$) and the untreated ($W_i=0$) customers and $X_i$ is the vector of observed customer features. According to Hitsch and Misra (2018), in an optimal policy, a customer is targeted if and only if $\mathbb{E}[\pi(1)|X_i]>\mathbb{E}[\pi(0)|X_i]$, which means that the expected profit in case of targeting exceeds the expected profit in case that the customer i is not targeted. This expression can be reformulated to: 

$\mathbb{E}[\pi(1)-\pi(0)|X_i] > 0$

$\Longleftrightarrow \mathbb{E}[(mY_i(1))-c)-Y_i(0)|X_i]$

$\Longleftrightarrow \mathbb{E}[Y_i(1)-Y_i(0)|X_i]-c >0$

$\Longleftrightarrow m\tau(x) < c$

Here, $\tau(x)$ is the conditional average treatment effect (CATE): $\tau(x)=E(Y_i(1)-Y_i(0) | X_i=x)$. The CATE describes the average causal effect of targeting for a sub-population of customers, with identical features. It is shown in Hitsch, Misra (2018), that the CATE is sufficient to identify an optimal targeting policy.  -->

## Estimating ${\tau}(x)$ directly via Causal KNN regression

As mentioned before, the true CATE is not observable. Hitsch and Misra use a method to approximate the actual treatment effect via the Causal KNN regression. This approach allows to derive an estimator of the CATE on an individual unit level. 
The Causal KNN regression method is represented in the following equation:


 <img src="http://chart.googleapis.com/chart?cht=tx&chl=$\hat{\tau}_K(x)=\frac{1}{K}\sum_{i\in N_K(x,1)}\cdot{Y_i}-\frac{1}{K}\sum_{i\in N_K(x,0)}\cdot{Y_i}$" style="border:none;">

 where $Y_i$ represents the outcome values of the target variable and $N_K(x,0)$ and $N_K(x,1)$ denote the nearest neighbour units with treatment status $W_i = 1$ and $W_i = 0$ respectively.
 Since the Causal KNN algorithm takes the number of nearest neighbours $K$ as a freely selectable parameter, it is necessary to find a way to choose the optimal $K$ value. In order to present the real data implementation of the Causal KNN regression, the used data will be presented in the following. 


### Data Presentation
For the application of the presented methodology, data from an E-Mail targeting campaign was used, including 64,000 observations. The data entries represent customer transactions with serveral corresponding variables. The treatment variable "segment" originally includes three possible values, E-Mail for Women, E-Mail for Men and No E-Mail. There are three different target variables. "Visit" as the visitation of the companies website and "Conversion" of the customer are binary variables. The numeric variable "Spend" represents the amount of money spend by a customer in the following two weeks after the campaign. 
There are various covariates that characterize customers according to their personal properties and purchasing behaviour. The "Recency" represents the month since the last purchase was made. "History" and "History_Segment" include information about the amount of money spend in the last year, in dollars. "Mens" and "Womens" are binary variables that indicate whether a customer bought products for men or for  women. The "Zip_Code" includes geographical information, grouped into Urban, Suburban and Rural regions. "Newbie" states whether a person was registered as a new customer in the past 12 month and the Channel represents on which ways a customer placed orders, i.e. via phone, mail or both. To provide a comprehensive overview about the data, an example table is presented, including first observations of the data set.

```r
head(data)
```

<!--
recency | history_segment | history | mens | womens | zip_code  | newbie | channel | segment       | visit | conversion | spend | idx   | treatment
------- | --------------- | ------- | ---- | ------ | --------- | ------ | ------- | ------------- | ----- | ---------- | ----- | ----- | ---------
10      | 2) $100 - $200  | 142.44  |  1   |   0    | Surburban |  0     | Phone   | Womens E-Mail | 0     | 0          | 0     | 1     | 1
6       | 3) $200 - $350  | 329.08  |  1   |   1    | Rural     |  1     | Web     | No E-Mail     | 0     | 0          | 0     | 2     | 0
7       | 2) $100 - $200  | 180.65  |  0   |   1    | Surburban |  1     | Web     | Womens E-Mail | 0     | 0          | 0     | 3     | 1
9       | 5) $500 - $750  | 675.83  |  1   |   0    | Rural     |  1     | Web     | Mens E-Mail   | 0     | 0          | 0     | 4     | 1
2       | 1) $0   - $100  | 45.34   |  1   |   0    | Urban     |  0     | Web     | Womens E-Mail | 0     | 0          | 0     | 5     | 1
6       | 2) $100 - $200  | 134.83  |  0   |   1    | Surburban |  0     | Phone   | Womens E-Mail | 1     | 0          | 0     | 6     | 1
-->

<style type="text/css" style="width:110%">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-zci2{font-weight:bold;background-color:#dae8fc;text-align:left;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-zci2">recency</th>
    <th class="tg-zci2">history_segment</th>
    <th class="tg-zci2">history</th>
    <th class="tg-zci2">mens</th>
    <th class="tg-zci2">womens</th>
    <th class="tg-zci2">zip_code</th>
    <th class="tg-zci2">newbie</th>
    <th class="tg-zci2">channel</th>
    <th class="tg-zci2">segment</th>
    <th class="tg-zci2">visit</th>
    <th class="tg-zci2">conversion</th>
    <th class="tg-zci2">spend</th>
    <th class="tg-zci2">idx</th>
    <th class="tg-zci2">treatment</th>
  </tr>
  <tr>
    <td class="tg-0lax">10</td>
    <td class="tg-0lax">2) 100-100-200</td>
    <td class="tg-0lax">142.44</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">Surburban</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">Phone</td>
    <td class="tg-0lax">Womens E-Mail</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">1</td>
  </tr>
  <tr>
    <td class="tg-0lax">6</td>
    <td class="tg-0lax">3) 200-200-350</td>
    <td class="tg-0lax">329.08</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">Rural</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">Web</td>
    <td class="tg-0lax">No E-Mail</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">2</td>
    <td class="tg-0lax">0</td>
  </tr>
  <tr>
    <td class="tg-0lax">7</td>
    <td class="tg-0lax">2) 100-100-200</td>
    <td class="tg-0lax">180.65</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">Surburban</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">Web</td>
    <td class="tg-0lax">Womens E-Mail</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">3</td>
    <td class="tg-0lax">1</td>
  </tr>
  <tr>
    <td class="tg-0lax">9</td>
    <td class="tg-0lax">5) 500-500-750</td>
    <td class="tg-0lax">675.83</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">Rural</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">Web</td>
    <td class="tg-0lax">Mens E-Mail</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">4</td>
    <td class="tg-0lax">1</td>
  </tr>
  <tr>
    <td class="tg-0lax">2</td>
    <td class="tg-0lax">1) 0-0-100</td>
    <td class="tg-0lax">45.34</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">Urban</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">Web</td>
    <td class="tg-0lax">Womens E-Mail</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">5</td>
    <td class="tg-0lax">1</td>
  </tr>
  <tr>
    <td class="tg-0lax">6</td>
    <td class="tg-0lax">2) 100-100-200</td>
    <td class="tg-0lax">134.83</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">Surburban</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">Phone</td>
    <td class="tg-0lax">Womens E-Mail</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">6</td>
    <td class="tg-0lax">1</td>
  </tr>
</table>


#### Data Preparation

After importing the e-mail data set, the non-numeric variables are converted into factors. Afterwards, all columns are examined with regard to missing values. There are no missing observations in the data set. Therefore, an imputation of missing values is not necessary for none of the variables.

To label the different observations in the data set, an index column was added, in order to be able to reassign all observations in the later steps of the analysis. The original data set contains three outcomes for the treatment variable, i.e. E-Mail for Women, E-Mail for Men and no E-Mail. Since the analysis is designed to deal with binary treatment variables, the three possible treatment values are reduced to either receiving an E-Mail or not. For the further process of the analysis, the binary variable "treatment" indicates solely whether units received a treatment. There is no distinction anymore between E-Mails for Women and Men. 

Since the different treatments were randomly assigned with the same probability, the aggregation of the treatment yields at a propensity score of 2/3 for each individual. Therefore, the treatment assignment in the data set is slightly unbalanced. Since the difference between the number of treated and untreated customers is not too high, the unbalancedness should not cause any problems for the proposed method. For extremely unbalanced data sets with regard to the treatment status, the Causal KNN algorithm might not work, because it can become impossible to find sufficiently much treated or untreated nearest neighbours.

The KNN algorithm identifies the nearest neighbours of an individual $i$ by calculating the euclidean distances between target variable outcomes of different units. Therefore, the method can only deal with numeric variables. To meet this requirement, the non-numeric covariates are transformed into dummy variables, using the mlr package.

<!-- ```{r, eval = FALSE, include = TRUE}
###preparation of the data set
#adding index column
data$idx = seq(1:nrow(data))

#adding binary treatment column
data$treatment = sapply(data$segment, FUN = function(x) ifelse(x == "No E-Mail", 0, 1))

#create dummy features
library(mlr)

data_d = createDummyFeatures(data, target = "visit")
colnames(data_d)
``` -->
<!-- $\hat{\tau}_K(x) =$ $\frac{1}{K}$ $\sum\limits_{i \in N_K(x,1)}$

$\widehat{\tau_K}(x) = \frac{1}{K} \sum_{i \in N_K(x,1)} Y_i - \frac{1}{K}  \sum_{i \in N_K(x,0)} Y_i$

$\sum\limits_{i \in N_K(x,1)}$

$\hat{\tau}_K(x)=\frac{1}{K} \sum\limits_{i \in N_K(x,1)} Y_i$

$\hat{\tau}_K(x)=\frac{1}{K}$ $\sum_{i \in N_K(x,1)} Y_i$ - $\frac{1}{K}$  $\sum_{i \in N_K(x,0)} Y_i$, where $Y_i$ -->

### Causal KNN Implementation

In the application here, the propensity score is the probability of receiving an email, i.e. the 'targeting probability' for each unit. The targeting probability is constant over all customers with a value of 2/3 in our application. That means, a randomly choosen customer receives an email with a probability of 2/3. 

The algorithm starts with selecting the k treated and untreated nearest neighbours for each observation. The nearest neighbours are selected based on the covariates, using the euclidean distance measure on the standardized data. However, every other distance measure would be similarily feasible, depending on the application framework. Afterwards, the individual treatemnt effects are derived by computing the differences between the mean outcome values of the treated and the untreated nearest neighbours for all units. This yields at an estimation of the CATE for each observation in the data set. A brief description of the algorithm is given by the following figure: 

<img align="center" width="800"
style="display:block;margin:0 auto;" 
src="/blog/img/seminar/causal_knn/C-KNNProcedure.png"
alt = "The Causal KNN Algorithm">



The Causal KNN alorithm starts with selecting the k treated and untreated nearest neighbour observations for each unit i. To allow for a simple selection of both groups, the data set is separated into two data frames, containing the treated and untreated observations respectively.

```{r, eval = FALSE, include = TRUE}
#select data partition for training
data_d = data_d[1:10000, ]

###splitting data set
#1 no treatment
data_nt = data_d[which(data_d$segment.No.E.Mail == 1),]

#2 treatment
data_t  = data_d[which(data_d$segment.No.E.Mail == 0),]

```

The KNN search is done by using the FNN package in R. This package allows to extract the index values of the nearest neighbours for each observation and the corresponding distances. An important aspect of Causal KNN is to find the optimal value of $K$ for a specific application setting. To check for several values of k, it is necessary to specify the maximum number of nearest neighbours, that should be tested. The maximum possible value of $K$ is restricted by the minimum total number of treated or untreated observations, i.e. the minimum number of rows of the data frames with the two treatment groups. 

A very useful feature of the FNN package is the possibility to specify different data sets for the search space and the observations for which nearest neighbours have to be found. This allows to conduct a simple search for nearest neighbours among the treated and untreated observations. There are two data frames resulting from both steps of the search process. The "treated_nn" data frame includes the nearest neighbours with treatment status $W_i=1$ and "untreated_nn" those with treatment status $W_i=0$. These two data sets build the basis for the next step of the algorithm, i.e. the identification of the optimal value of $K$.

```{r, eval = FALSE, include = TRUE}
###Causal KNN
#install.packages("FNN")
library(FNN)

#setting parameter k for the maximum number of nearest neighbours to test for
k = 3000
k = k + 1

#select columns to be eliminated for nearest neighbour search
drop_cols = c("visit", 
              "spend", 
              "conversion", 
              "idx", 
              "treatment", 
              "segment.Mens.E.Mail",
              "segment.Womens.E.Mail",
              "segment.No.E.Mail")

#garbage collection to maximize available main memory
gc()

#calculate indices of k treated nearest neighbours
treated_nn = get.knnx(data_t[, !(names(data_t) %in% drop_cols)], 
                      query = data_d[, !(names(data_d) %in% drop_cols)], 
                      k = k)
treated_nn = data.frame(treated_nn$nn.index)

#deleting first column
treated_nn = treated_nn[, 2:ncol(treated_nn)]


#calculate indices of k untreated nearest neighbours
untreated_nn = get.knnx(data_nt[, !(names(data_nt) %in% drop_cols)], 
                        query = data_d[, !(names(data_d) %in% drop_cols)], 
                        k = k)
untreated_nn = data.frame(untreated_nn$nn.index)

#deleting first column
untreated_nn = untreated_nn[, 2:ncol(untreated_nn)]


#replacing index values
treated_nn = sapply(treated_nn, FUN = function(x){
  data_t$idx[x]
})

untreated_nn = sapply(untreated_nn, FUN = function(x){
  data_nt$idx[x]
}) 

#transpose data frames
treated_nn   = t(treated_nn)
untreated_nn = t(untreated_nn)
```

<!--
$\hat{\tau}_K(x)=\frac{1}{K} \sum_{i \in N_K(x,1)} Y_i - \frac{1}{K}  \sum_{i \in N_K(x,0)} Y_i$
-->

<!-- These methods select the tuning parameter K using a feasible loss that, in expectation, attains its minimum at the same K value as the infeasible loss based on the treatment effect. 
For any vector x, we find the K nearest *treated* neighbors and separately the K nearest *untreated* neighbors. We then estimate the CATE using the difference between the nearest treated and untreated units:
$\widehat{\tau_K}(x)= \frac{1}{K} \sum_{i \in N_K(x,1)} Y_i - \frac{1}{K}  \sum_{i \in N_K(x,0)} Y_i$.-->

                                                                                                       
### Comparison of Causal KNN and the Two-Model-Approach
At this point, a distinction between the Causal KNN and the Two-Model-Approach has to be made, since both methods seem to be similar. According to Rzepakowski and Jaroszewicz (2012, 2), the idea behind the two model approach is to build two separate models, to estimate the treatment effect. One Model is trained, using the treatment, and the other one using the control data set. After building the two models, the treatment effect estimations are calculated by subtracting the predicted class probabilities of the control model from those of the treatment model. This yields at a direct estimation of the difference in the outcome, caused by the treatment. An advantage of the two model approach is that it can be applied with any classification model (Rzepakowski and Jaroszewicz, 2012). A possible problem that might arise with this approach, is that the uplift might be different from the class distributions. In this case, the both models focus on predicting the class, rather than the treatment effect.
The Causal KNN ALgorithm also calculates the difference between the predicted outcomes of the treated and untreated observations in a data set. The number of observations that are taken into account, is limited by the KNN parameter K. Therefore, the Causal KNN algorithm itself can be possibly seen as a special case of the two model approach. An important aspect of the described method here, is the usage of the transformed outcome for the tuning of the Causal KNN model. Since the combination of Causal KNN and the transformed outcome allows to calculate the CATE estimations directly, it is not appropriate to completely assign this to the group of two model approaches. According to Rzepakowski and Jaroszewicz (2012, 2), the two model approach is different from the direct estimation methods, since it focuses on predicting the outcome of the target variable to estimate the treatment effect, rather than predicting the treatment effect itself. As already shown above, the Causal KNN algorithm should be rather categorized as a direct estimation approach (Hitsch and Misra, 2018) and should therefore not be seen as a pure two model approach.

To estimate a robust CATE, we would likely use all customers in the neighborhood. A high value for K increases the accounted neighborhood, while it decreases the similarity between the customers. Therefore, the estimation depends on the choice of K. To find an optimal value of K, we want to minimize the squared difference between $\hat{\tau}_K(x)$ and the transformed outcome $Y^*$. This is called the **Transformed Outcome Approach**, which will be described in the following.

## Transformed Outcome Approach

Naturally, the optimal value of $K$ is given for the minimum of the transformed outcome loss: 

$\mathbb{E}[(\tau_k(X_i)-\widehat{\tau_k}(X_i))^2]$

Unfortunately, as mentioned before, this criterion is infeasible, since the true CATE $\tau_K(X_i)$ is not observable. The transformed outcome approach is a method to somehow overcome the described fundamental problem of Causal inference by estimating the true CATE. In the literature, we find multiple approaches to do so. Gutierrez (2017) presents three different methods. First,he mentions the two-model approach that trains two models, one for the treatment group and one for the control group. Second, he mentions the class transformation method that was first introduced by Jaskowski and Jaroszewicz (2012). Their approach works for binary outcome variables (i.e. $Y_i^{obs.}=\text{{0,1}}$) and a balanced dataset (i.e. $e(X_i)=\frac{1}{2}$). A general formulation for the class transformation, which is even applicable for numeric outcome variables and an unbalanced dataset is given by Athey and Imbens (2015b). The transformed outcome is defined as follows: 

$Y_i^*=W_i\cdot\frac{Y_i(1)}{e(X_i)}-(1-W_i)\cdot\frac{Y_i(0)}{1-e(X_i)}$

We can rewrite this formula:

$Y_i^*=\frac{W_i - e(X_i)}{e(X_i)(1-e(X_i))}\cdot Y_i^{obs.}$

The latter formula allows us to calculate the transformed outcome with the observed outcome. In case the propensity score is constant for every individual, i.e. complete randomization, the formula simplifies to $Y_i^*=Y_i^{obs}/e(x)$ for the treated individuals and $-Y_i^{obs}/(1-e(x))$ for the individuals that were not treated (c.f. Athey and Imbens 2015b).

 Now, why is the transformed outcome helpful? In the following, we will present two appealing properties of the transformed outcome.

First, with the required assumption of unconfoundedness and no overlap, it can be shown that the transformed outcome is an unbiased estimator for the CATE: 

$Y_i^*=W_i\cdot\frac{Y_i(1)}{e(X_i)}-(1-W_i)\cdot\frac{Y_i(0)}{1-e(X_i)}\quad|\quad\mathbb{E}[\cdot|W_i,X_i=x]$

$\Longrightarrow\mathbb{E}[Y_i^*]=\mathbb{E}[W_i|X_i=x]\cdot\frac{\mathbb{E}[Y_i|W_i=1,X_i=x]}{e(X_i)}-\mathbb{E}[1-W_i|X_i=x]\cdot\frac{\mathbb{E}[Y_i|W_i=0,X_i=x]}{1-e(X_i)}$

$=e(X_i)\cdot\frac{\mathbb{E}[Y_i|W_i=1,X_i=x]}{e(X_i)}-(1-e(X_i))\cdot\frac{\mathbb{E}[Y_i|W_i=0,X_i=x]}{1-e(X_i)}$

$=\mathbb{E}[Y_i|W_i=1,X_i=x]-\mathbb{E}[Y_i|W_i=0,X_i=x]\quad|\quad\textbf{Unconfoundedness: $Y_i(0), Y_i(1) \bot W_i | X_i$ }$

$=\mathbb{E}[Y_i(1)-Y_i(0)|X_i=x]$

$=\tau(X)$

If the overlap assumption is violated, it would not be possible to determine $\mathbb{E}[Y_i|W_i=1, X_i=x]$ and $\mathbb{E}[Y_i|W_i=0, X_i=x]$, because there would be only treated or untreated individuals for specific values of $x$. Since the estimator $Y_i^*=\tau_K(X_i)+\nu_i$ is unbiased, the error term $\nu_i$ is orthogonal to any function of $X_i$ (i.e. exogeneity assumption $\mathbb{E}[\nu_i|X_i]=0$). 

The second appealing property of the transformed outcome is then demonstrated by Hitsch and Misra (2018): 

$\mathbb{E}[(Y_i^*-\widehat{\tau_K}(X_i))^2|X_i] = \mathbb{E}[({\tau_K}(X_i)+\nu_i-\widehat{\tau_K}(X_i))^2|X_i]$

$=\mathbb{E}[({\tau_K}(X_i)-\widehat{\tau_K}(X_i))^2+2\cdot({\tau_K}(X_i)-\widehat{\tau_K}(X_i))\cdot\nu_i+\nu_i^2|X_i]$

$=\mathbb{E}[({\tau_K}(X_i)-\widehat{\tau_K}(X_i))^2|X_i]+\mathbb{E}[\nu_i^2|X_i]$. 

$\Longrightarrow\mathbb{E}[(Y_i^*-\widehat{\tau_K}(X_i))^2]=\mathbb{E}[({\tau_K}(X_i)+\nu_i-\widehat{\tau_K}(X_i))^2)]=\mathbb{E}[({\tau_K}(X_i)-\widehat{\tau_K}(X_i))^2]+\mathbb{E}[\nu_i^2]$

In equation three we used the exogeneity assumption as well as the linearity of the expected value operator. 
As we can see,  $\mathbb{E}[\nu_i^2]$ does not depend on the value of K, so minimization of the transformed outcome loss also minimizes the infeasible outcome loss. 
Therefore the transformed outcome allows us to estimate the true CATE and with this, we can find an optimal value for the parameter k. 
 
## Parameter Tuning using the Transformed Outcome

An essential part of the Causal KNN algorithm is the parameter tuning to specify the value of $K$, that leads to the most accurate treatment effect estimations. For the parameter tuning, several values for $K$ are tested and different CATE estimations are stored in a separate data frame. To derive estimations for all observations in the data set, the differences of the mean outcome variable values, of the k treated and untreated nearest neigbours, are calculated. Several iterations of this precedure yield at a data frame, containing the individual uplift estimations for different values of k. To select the most accurate estimation, the transformed outcome is used to decide for a preferable k value.

```{r, eval = FALSE, include = TRUE}
###parameter tuning to find optimal k value
#setting parameters for the number of neigbours
k_start = 50
k_end   = 3000
steps   = 50

#creating sequence of k values to test for
k_values = seq(from = k_start, to = k_end, by = steps)

#preparing uplift data frame
uplift = data.frame("idx" = data_d$idx,
                    "treatment" = data_d$treatment)

#calculating uplift for specified k values
for (k in k_values) {
  reaction_nt = apply(untreated_nn, MARGIN = 2, 
                      FUN = function(x){
                        mean(data_d$visit[x[1:k]])
                      }
  )
  
  reaction_t = apply(treated_nn, MARGIN = 2, 
                     FUN = function(x){
                       mean(data_d$visit[x[1:k]])
                     }
  )
  
  uplift[paste("uplift_",k, sep = "")] = reaction_t - reaction_nt
  
  print(paste("k = ", k))
}
```

## Transformed Outcome Loss Calculation

After calculating the various uplift estimations by varying the tuning parameter $K$ the transformed outcome as a proxy for the true CATE, is constructed. The calculation of the transformed outcome requires the knowledge of the propensity score, the treatment status and the original value of the target variable as inputs. By using the formula introduced above, the transformed outcome can be easily calculated for the actual target variabe visit. To decide on a value of $K$, the transformed outcome loss is calculated for each potential $K$ value by computing the mean squared difference between the transformed outcome and the various treatment effect estimations. The estimation, that yields at the minimal value of the transformed outcome loss, indicates the most accurate value for $K$, as shown by Hitsch and Misra (2018). The following plot was created to depict the development of the outcome loss with varying values of $K$. When increasing the number of nearest neighbours that are taken into consideration, the accuracy of the CATE estimation increases in the beginning. However, when increasing $K$ over a certain level, the heterogeniety among the nearest neighbours increases too much. That leads to less accurate estimations of the treatment effects. Therefore the aim is to find an optimal number of nearest neighbours, where sufficiently observations are taken into account, without increasing the heterogeniety too much.

```{r, eval = FALSE, include = TRUE}
###transformed outcome

#Identify/Set the propensity score

e_x = mean(data_d$treatment)

#function for transformed outcome
transformed_outcome = function(e_x, w_i, target){
  return(((w_i - e_x)/(e_x*(1-e_x)))*target)
}

#apply function on all observations
trans_out = sapply(uplift$idx, FUN = function(x){
  transformed_outcome(e_x = e_x, 
                      w_i = data$treatment[x], 
                      target = data$visit[x])
})

uplift$trans_out = trans_out

#transformed outcome loss
outcome_loss = data.frame("k" = k_values, "loss" = 0)

#find optimal k value from transformed outcome loss
for (i in 1:length(k_values)){
  outcome_loss[i, 2] = mean((uplift$trans_out - 
                               uplift[, i + 2])^2)
}

#find minimal outcome loss value
outcome_loss$k[which.min(outcome_loss$loss)]
```

<!-- #plot result
library(ggplot2)

k_plot = ggplot(data = outcome_loss) +
  geom_point(aes(x = outcome_loss$k, 
                 y = outcome_loss$loss), 
             size = 2, shape = 18) +
  geom_point(aes(x = outcome_loss$k[which.min(outcome_loss$loss)], 
                 y = min(outcome_loss$loss)), 
             size = 4, shape = 18, color = "red") +
  geom_text(aes(x = outcome_loss$k[which.min(outcome_loss$loss)], 
                y = min(outcome_loss$loss)), 
            label = paste("K = ", outcome_loss$k[which.min(outcome_loss$loss)]), 
            color = "black", size = 4, nudge_x = 120, check_overlap = TRUE) +
  stat_smooth(data = outcome_loss, 
              aes(x = outcome_loss$k, y = outcome_loss$loss), 
              method = "loess", se = FALSE, span = 0.1) +
  labs(title="Parameter Optimization of Causal-KNN CATE-Estimation", 
       x ="Value of K", y = "Outcome Loss Value") +
  theme_light()

k_plot -->


<img align="center" width="900"
style="display:block;margin:0 auto;" 
src="/blog/img/seminar/causal_knn/k_plot.png"
alt = "Outcome Loss for Different Values of K">

After finding the most accurate estimation of the CATE, it is possible to extract valuable insights from the data. This information can be used to plan targeting policies for future campaigns. Since the Causal KNN algorithm delivers treatment effect estimation on an individual level, it is easily possible to extract several other treatemnt effect measures. Furthermore, different CATE estimations based on different dimensions of the data can be calculated by grouping the observations.

<!-- ```{r, eval = FALSE, include = TRUE}
#extract best uplift estimation
data_d$uplift = uplift[, paste0("uplift_", outcome_loss$k[which.min(outcome_loss$loss)])]
data_d$uplift[1:20]

#CATE
cate_sex = aggregate(data_d$uplift, by=list(data$mens[1:nrow(data_d)]), FUN=mean)
cate_sex

cate_hist_seg = aggregate(data_d$uplift, by=list(data$history_segment[1:nrow(data_d)]), FUN=mean)
cate_hist_seg

ggplot(data = cate_hist_seg, aes(x = Group.1, y = x)) +
  geom_bar(stat = "identity", fill = "#adc7db") +
  labs(title="CATE-Estimation for different History Segments", x ="History Segment", y = "CATE") +
  theme_light() +
  geom_text(aes(label = round(x, digits = 4)), vjust = 1.6, color = "black", size = 3.5)

cate_zip_code = aggregate(data_d$uplift, by=list(data$zip_code[1:nrow(data_d)]), FUN=mean)
cate_zip_code

cate_newbie = aggregate(data_d$uplift, by=list(data$newbie[1:nrow(data_d)]), FUN=mean)
cate_newbie

ggplot(data = cate_newbie, aes(x = Group.1, y = x)) +
  geom_bar(stat = "identity", fill = "#adc7db") +
  labs(title="CATE-Estimation for different Newbie-Status", x ="Newbie", y = "CATE") +
  theme_light() +
  geom_text(aes(label = round(x, digits = 4)), vjust = 1.6, color = "black", size = 3.5)

cate_channel = aggregate(data_d$uplift, by=list(data$channel[1:nrow(data_d)]), FUN=mean)
cate_channel

ggplot(data = cate_channel, aes(x = Group.1, y = x)) +
  geom_bar(stat = "identity", fill = "#adc7db") +
  labs(title="CATE-Estimation for different Communication Channels", x ="Communication Channel", y = "CATE") +
  theme_light() +
  geom_text(aes(label = round(x, digits = 4)), vjust = 1.6, color = "black", size = 3.5)
``` -->

<img align="center" width="800"
style="display:block;margin:0 auto;" 
src="/blog/img/seminar/causal_knn/cate-hist_seg.png"
alt = "CATE for History Segments">

<img align="center" width="800"
style="display:block;margin:0 auto;" 
src="/blog/img/seminar/causal_knn/cate_chann.png"
alt = "CATE for different contacting Channels">


Another advantage of the Causal KNN method is the high flexibility with regard to the target variable. As already shown, the algorithm works with binary outcome variables, i.e. visit in this case. By simply exchanging the target variable with a numeric one ("spend"), the algorithm can be executed similarily, without further modifications to the code. The CATE estimations represent an approximation of the total uplift, that is expected in the case of a treatment. Therefore, the CATE for the target variable "spend" indicates the expected absolute increase or deacrease of spendings for each individual, when a treatment is assigned. The application of the Causal KNN algorithm for numeric target variables does not need further changes to the implementation, instead of changing the outcome variable from visit to spend. Therefore, the code is not provided here in the blogpost. A possible implementation can be seen here: (https://github.com/mkrcke/APA/blob/master/src/07_causal_knn_numeric.R). Although the data theoretically allows for an estimation of the CATE for the variable "spend", the insights of that estimation are not that valuable, because the dataset is extremely unbalanced with regard to the actual conversions (i.e. "spend" $> 0$). 

### Causal KNN for optimal Targeting Policy Prediction

The next section deals with the Causal KNN method in an application case, where a targeting policy has to be developed, based on the CATE estimations. From the model training and parameter tuning of previous steps, it was possible to find a best fitting value for the $K$ nearest neighbours that are taken into account. This value is now used in the prediction setting, where we assume that we do not know the actual outcome value. 

The aim is to find those individuals, where a treatment increases the value of the outcome variable the most. By specifying a fixed percentage of the population, we are able to select the most profitable individuals for a treatment assignment. For the test data, we assume that we did not decide for a treatment assignment yet and do not have any information about the individual outcome values. Therefore, we are not able to calculate the transformed outcome for the test data set. Since we already estimated optimal parameter of k, we use this value here as parameter for the Causal KNN algorithm. The difference is now that we do not search for treated and untreated nearest neighbours in the test data set. We are not able to do so, because we did not decide for any treatment assignment yet. We rather search the neighbours for the observations in the training data, which we already used for finding the optimal k value. 

```{r, eval = FALSE, include = TRUE}
test_set = createDummyFeatures(data[25001:30000, ])
test_set$idx = 1:nrow(test_set)

#splitting data set
#1 no treatment
data_nt = data_d[which(data_d$segment.No.E.Mail == 1),]

#2 treatment
data_t = data_d[which(data_d$segment.No.E.Mail == 0),]


#running Causal KNN for test set to calculate MSE
#select target columns
drop_cols = c("visit", 
              "spend", 
              "conversion", 
              "idx", 
              "treatment", 
              "segment.Mens.E.Mail",
              "segment.Womens.E.Mail",
              "segment.No.E.Mail",
              "uplift")

#setting optimal k value from the parameter tuning above
k = outcome_loss$k[which.min(outcome_loss$loss)] + 1


#calculate indices of k treated nearest neighbours
treated_nn = get.knnx(data_t[, !(names(data_t) %in% drop_cols)], 
                      query = test_set[, !(names(test_set) %in% drop_cols)], 
                      k = k)
treated_nn = data.frame(treated_nn$nn.index)

#deleting first column
treated_nn = treated_nn[, 2:ncol(treated_nn)]


#calculate indices of k untreated nearest neighbours
untreated_nn = get.knnx(data_nt[, !(names(data_nt) %in% drop_cols)], 
                        query = test_set[, !(names(test_set) %in% drop_cols)], 
                        k = k)
untreated_nn = data.frame(untreated_nn$nn.index)

#deleting first column
untreated_nn = untreated_nn[, 2:ncol(untreated_nn)]

#replacing index values
treated_nn = sapply(treated_nn, FUN = function(x){
  data_t$idx[x]
})

untreated_nn = sapply(untreated_nn, FUN = function(x){
  data_nt$idx[x]
}) 

#transpose data frames
treated_nn = t(treated_nn)
untreated_nn = t(untreated_nn)

#preparing uplift data frame
uplift = data.frame("idx" = test_set$idx,
                    "treatment" = test_set$treatment)

reaction_nt = apply(untreated_nn, MARGIN = 2, FUN = function(x){
  mean(data_d$visit[x[1:k-1]])
})

reaction_t = apply(treated_nn, MARGIN = 2, FUN = function(x){
  mean(data_d$visit[x[1:k-1]])
})

uplift[paste("uplift_", k, sep = "")] = reaction_t - reaction_nt

```

## Application of Transformed Outcome Approach for general Uplift Models

### Causal Tree Model
The parameter tuning using the transformed outcome loss is equally useful for other uplift applications. To show this, the same methodology that was applied to determine the optimal $K$ value for the Causal KNN model, is used here to find optimal parameters for a causal tree. Different values for parameters are tested and evaluated by using the transformed outcome loss. Again, the minimal outcome loss value indicates the most favorable parameter value for the model estimation.

```{r, eval = FALSE, include = TRUE}
#using transformed outcome for tuning parameters of other uplift models
#different cp for causal tree

#install.packages("devtools")
#library(devtools) 
#install_github("susanathey/causalTree")

library(causalTree)

#setting parameters for the complexity parameters to test for
cp_start = 0
cp_end   = 0.0001
cp_steps = 0.00001

#creating sequence of k values to test for
cp_values = seq(from = cp_start, to = cp_end, by = cp_steps)
length(cp_values)

uplift_ct = data.frame("idx" = data_d$idx, "treatment" = data_d$treatment, "visit" = data_d$visit) 

#calculating uplift for specified min size values
for (cp in cp_values) {
  causal_tree = causalTree(visit~.-spend -conversion -idx -segment.Mens.E.Mail -segment.Womens.E.Mail -segment.No.E.Mail, data = data_d, treatment = data_d$treatment, split.Rule = "CT", cv.option = "CT", split.Honest = T, cv.Honest = T, split.Bucket = F, xval = 5, cp = cp, propensity = e_x)
  
  uplift_ct[paste("uplift_",cp, sep = "")] = sapply(causal_tree$where, FUN = function(x){causal_tree$frame$yval[x]})
  
  print(paste("cp =", cp))
}


#calculate the transformed outcome
uplift_ct$trans_out = sapply(uplift_ct$idx, FUN = function(x){
  transformed_outcome(e_x = e_x, w_i = data_d$treatment[x], target = data_d$visit[x])
})

#transformed outcome loss
outcome_loss = data.frame("cp" = cp_values, "loss" = 0)

#find optimal cp value
for (i in 1:length(cp_values)){
  outcome_loss[i, 2] = mean((uplift_ct$trans_out - uplift_ct[, i + 3])^2)
}

outcome_loss

#find minimal value
min(outcome_loss$loss)

outcome_loss$cp[which.min(outcome_loss$loss)]

plot(outcome_loss$loss)

cp_plot = ggplot(data = outcome_loss) +
  geom_point(aes(x = outcome_loss$cp, y = outcome_loss$loss), size = 2, shape = 18) +
  geom_point(aes(x = outcome_loss$cp[which.min(outcome_loss$loss)], y = min(outcome_loss$loss)), 
             size = 4, shape = 18, color = "red") +
  geom_text(aes(x = outcome_loss$cp[which.min(outcome_loss$loss)], y = min(outcome_loss$loss)), 
            label = paste("cp = ", outcome_loss$cp[which.min(outcome_loss$loss)]), color = "black", size = 4, 
            nudge_x = 0.00001, nudge_y = 0, check_overlap = TRUE) +
  stat_smooth(data = outcome_loss, aes(x = outcome_loss$cp, y = outcome_loss$loss), method = "loess", 
              se = FALSE, span = 0.1) +
  labs(title="Parameter Optimization of Causal-KNN CATE-Estimation", x ="Value of Complexity Parameter", y = "Outcome Loss Value") +
  theme_light()

cp_plot
```

<img align="center" width="800"
style="display:block;margin:0 auto;" 
src="/blog/img/seminar/causal_knn/cpplot.png"
alt = "Outcome Loss for Different Complexity Parameters of the Causal Tree Model">


To compare the results of the causal KNN model with the estimations of another uplift model, a causal tree was learned, using the implementation of Susan Athey. The optimal complexity parameter (cp) from the tuning part is used for the CATE predictions of the causal tree.

```{r, eval = FALSE, include = TRUE}
### building causal tree model for test set
#extracting optimal vlaue for complexity parameter (cp) from tuning
optimal_cp = outcome_loss$cp[which.min(outcome_loss$loss)]

#learning causal tree model for the test set
causal_tree = causalTree(visit~.-spend -conversion -idx -segment.Mens.E.Mail -segment.Womens.E.Mail -segment.No.E.Mail, 
                         data = test_set, treatment = test_set$treatment, split.Rule = "CT", cv.option = "CT", 
                         split.Honest = T, cv.Honest = T, split.Bucket = F, xval = 5, cp = optimal_cp, minsize = 20, 
                         propensity = e_x)

uplift_ct = data.frame("idx" = test_set$idx, "treatment" = test_set$treatment, "visit" = test_set$visit) 

uplift_ct$uplift = sapply(causal_tree$where, FUN = function(x){causal_tree$frame$yval[x]})
```


## Model Evaluation


### MSE

To find the most promising individuals to target, the uplift data frame is sorted according to the individual CATE estimations. After specifying the percentage of individuals to receive a treatment, it is possible to select the first $x$% observations from the sorted data frame. For these individuals, a treatment is expected to have the highest impact on the outcome variable. To evaluate the model, Hitsch and Misra (2018) propose the mean squared error as an indicator of the model quality. This measure also allows to compare the model with the proposed treatment assignment of other uplift models. It is shown by Hitsch and Misra (2018) that, under unconfoundedness, the mean squared error based on the transformed outcome, is applicable for the comparison of different model fits on a particular data set. Therefore, the transformed outcome can be used for model evaluation as well (c.f. Hitsch & Misra, p.27, 2018). We can use the transformed outcome in the same manner as we used it when we tuned the parameter k. The criterion to minimize is therefore given by:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=$MSE_b=\frac{1}{|\tau_b|}\sum\limits_{i\in{\tau_b}}(Y_i^*-\widehat{\tau_b}(X_i))^2$" style="border:none;">


```{r, eval = FALSE, include = TRUE}
```

## Qini Curves for evaluation of the causal KNN model

The Gini-Coefficient is a widely used metric to compare the fit of different modeling approaches on a particular data set. The metric is intuitive and applicable, since it provides a single value for the evaluation. The value of the Gini-Coefficient is defined between 0 and 1, where 1 represents a perfect model and 0 indicates a fit that is not performing any better than a random ranking of customers. Devriendt et al. (2018) state that the Gini-Coefficient, however, is not readily applicable in uplift modeling context, since for every class we have individuals from the treatment and the control group. Therefore, in this context, the so called "Qini-Coefficient" is preferred. The Qini-Coefficient is mathematically equal to the difference of the Gini-Curve for the treatment group and the control group. 
The Qini-Coefficient provides an opportunity to evaluate uplift models according to the treatment effect estmations. Since the Qini is a common indicator for the quality of uplift models, it is used here to evaluate the Causal KNN model results. Based on the calculated uplift factors for every individual, one could target only those individuals with the highest uplift, i.e. the highest probability of visiting the website after receiving an email. Treatment assignment based on the uplift values can be compared to the random assignment, where every individual gets a treatment with a probability of 2/3. Finally, one can look at the cumulative visits for the targeted customers. If the targeting of customers with a higher uplift also corresponds to a higher chance of visit, the curve of the Causal KNN predictions must be above the curve of the randomized predictions. 
We can use the same procedure to evaluate the predictions of the Causal Tree Model. The following to graphs are illustrating the desired result. If we first target the customers we identified using the two model approaches, we will gain more desired outcomes (i.e. visits) than for the randomized targeting.


<img align="center" width="800"
style="display:block;margin:0 auto;" 
src="/blog/img/seminar/causal_knn/qini_knn.png"
alt = "Qini Curves for the Causal KNN Targeting Policy">

<img align="center" width="800"
style="display:block;margin:0 auto;" 
src="/blog/img/seminar/causal_knn/qini_ct.png"
alt = "Qini Curves for the Causal Tree Targeting Policy">   

  
### AUUC

The AUUC is a common measure for model evaluation. This measure is defined by the area under the uplift curve. We can compare the resulting uplift curves to the optimal curve. Therefore it has strong relations to the Qini-Curve, but this measure does not take the incremental effect of random targeting into account. The following table contains measures to compare the two applied uplift modeling approaches. 

    

<!-- Model                | MSE    | AUUC
-------------------- | ------ | -----------
Causal KNN (K = 450) | 0.5508 | 0.34
Causal Tree          | 0.5321 | 0.36 -->


## Results

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-cly1{text-align:left;vertical-align:middle}
.tg .tg-hjhm{font-weight:bold;background-color:#dae8fc;text-align:left;vertical-align:middle}
</style>
<table class="tg">
  <tr>
    <th class="tg-hjhm">Model</th>
    <th class="tg-hjhm">MSE</th>
    <th class="tg-hjhm">AUUC</th>
  </tr>
  <tr>
    <td class="tg-cly1">Causal KNN (K = 450)</td>
    <td class="tg-cly1"> 0.5508 </td>
    <td class="tg-cly1">0.34</td>
  </tr>
  <tr>
    <td class="tg-cly1">Causal Tree</td>
    <td class="tg-cly1">0.5321</td>
    <td class="tg-cly1">0.36</td>
  </tr>
</table>

<br>

 When comparing both, the causal KNN and the causal tree model, it can be observed that the causal tree performs better than the Causal KNN model. This holds true for evaluating both models in terms of the MSE as well as for the area under the uplift curves (AUUC).

## Conclusion

In this blogpost, we presented a method for estimating the conditional average treatment effect on an individual basis. This approach is called causal KNN. The causal KNN algorithm was implemented in R and applied to a real world data set from a randomized E-Mail marketing campaign. Furthermore, the transformed outcome was introduced, which represents the value of the "true" CATE in expectation, if several required assumptions to the data are fulfilled. The transformed outcome approach could be used for finding the optimal k value for the causal KNN model and also for the parameter tuning of other uplift modelling techniques. We trained a causal tree model to compare the results of our causal KNN estimations. The transformed outcome allowed for an evaluation of uplift models in terms of the MSE. Here, the causal tree performed better than the causal KNN model in terms of the MSE and the AUUC. Therefore, the causal KNN approach is a simple algorithm that delivers estimations of the CATE in uplift applications. However, the accuracy of the estimations is worse than those of comparable uplift models.


## Code distribution

The full source Code can be found here: https://github.com/mkrcke/APA


## References

Devriendt, F., Moldovan, D., & Verbeke, W. (2018). A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: A stepping stone toward the development of prescriptive analytics. Big data, 6(1), 13-41.

Gutierrez, P., & Gerardy, J. Y. (2017, July). Causal inference and uplift modelling: A review of the literature. In International Conference on Predictive Applications and APIs (pp. 1-13).

Pearl, J. (2009). Causal inference in statistics: An overview. Statistics surveys, 3, 96-146.

Zhou, X., & Kosorok, M. R. (2017). Causal nearest neighbor rules for optimal treatment regimes. arXiv preprint arXiv:1711.08451.

Dudik, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601.

Angrist, J. D. (2016). Treatment effect. The new Palgrave dictionary of economics, 1-8.

Jaskowski, M., & Jaroszewicz, S. (2012, June). Uplift modeling for clinical trial data. In ICML Workshop on Clinical Data Analysis.

Gubela, R. M., Lessmann, S., Haupt, J., Baumann, A., Radmer, T., & Gebert, F. (2017). Revenue Uplift Modeling.

Lo, V. S. (2002). The true lift model: a novel data mining approach to response modeling in database marketing. ACM SIGKDD Explorations Newsletter, 4(2), 78-86.

Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. Proceedings of the National Academy of Sciences, 113(27), 7353-7360.

Coussement, K., Harrigan, P., & Benoit, D. F. (2015). Improving direct mail targeting through customer response modeling. Expert Systems with Applications, 42(22), 8403-8412.

Hitsch, G. J., & Misra, S. (2018). Heterogeneous treatment effects and optimal targeting policy evaluation. Available at SSRN 3111957.

Ascarza, E., Fader, P. S., & Hardie, B. G. (2017). Marketing models for the customer-centric firm. In Handbook of marketing decision models (pp. 297-329). Springer, Cham.

Paul R. Rosenbaum; Donald B. Rubin, The Central Role of the Propensity Score in Observational Studies for Causal
Effects, Biometrika, Vol. 70, No. 1. (Apr., 1983), pp. 41-55.

Rzepakowski, P., & Jaroszewicz, S. (2012). Uplift modeling in direct marketing. Journal of Telecommunications and Information Technology, 43-50.

Athey, S., & Imbens, G. W. (2015). Machine learning methods for estimating heterogeneous causal effects. stat, 1050(5), 1-26.

Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. arXiv preprint arXiv:1707.00102.

Breiman, L. (1996). Bagging predictors. Machine learning, 24(2), 123-140.



## Package References

H. Wickham. ggplot2: Elegant Graphics for Data Analysis. Springer-Verlag New
  York, 2016.
  
Hadley Wickham, Jim Hester and Romain Francois (2018). readr: Read
  Rectangular Text Data. R package version 1.3.0.
  https://CRAN.R-project.org/package=readr
  
Bischl B, Lang M, Kotthoff L, Schiffner J, Richter J, Studerus E, Casalicchio
  G, Jones Z (2016). mlr: Machine Learning in R. _Journal of Machine Learning
  Research_, *17*(170), 1-5. <URL: http://jmlr.org/papers/v17/15-066.html>

Alina Beygelzimer, Sham Kakadet, John Langford, Sunil Arya, David Mount and
  Shengqiao Li (2019). FNN: Fast Nearest Neighbor Search Algorithms and
  Applications. R package version 1.1.3. https://CRAN.R-project.org/package=FNN
  
Hadley Wickham, Jim Hester and Winston Chang (2019). devtools: Tools to Make
  Developing R Packages Easier. R package version 2.0.2.
  https://CRAN.R-project.org/package=devtools
  
Susan Athey, Guido Imbens and Yanyang Kong (2016). causalTree: Recursive
  Partitioning Causal Trees. R package version 0.0.

