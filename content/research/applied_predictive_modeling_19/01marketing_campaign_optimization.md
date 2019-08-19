+++
title = "Marketing Campaign Optimization: Profit modeling"
date = '2019-08-12'
tags = [ "Causal Inference", "Class19",]
categories = ["Course projects"]
banner = "img/seminar/nn_fundamentals/neuralNetworkKlein.jpg"
author = "Seminar Applied Predictive Modeling (SS19) – Asmir Muminovic, Lukas Kolbe"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = " "

+++


# Marketing Campaign Optimization

<br>
## Causal Inference in Profit Uplift Modeling
#### Authors: Asmir Muminovic, Lukas Kolbe

<br>
### Motivation

The global spending on advertising amounts to more than 540 billion US dollars for 2018 only, and the spending for 2019 is predicted to reach over 560 billion US dollars. The marketing spendings have continuously increased since 2010 [17], and this trend does not seem to brittle in the near future.
Although marketing is such an integral part of most businesses, marketers often fail to maximize the profitability of their marketing efforts.
Traditionally, this issue was tackled via utilization of so-called “Response” models [18], which targets the most likely to buy customers. This model has a big downside, since it disregard causality. We effectively do not know if the targeted customers would have bought anyways, is annoyed by the targeting, or does not react at all. Targeting these customers would result in unnecessary costs. 
We should therefore select customers who will buy because of the campaign, that is, those who are likely to buy if targeted, but unlikely to buy otherwise [1]. This could be achieved with uplift modeling, since it measures the treatment effect. Additionally, customers have different spendings, therefore revenue uplift modeling seems more viable than standard conversion modeling. But, since targeting customers can come with a price in form of vouchers which effectively reduce the profits, Profit Uplift Modeling is what truly optimizes the profitability of online marketing campaigns.

<br>
### Datasets
For this project, we used four different datasets. All of them contain data from marketing campaigns in online shops, each having a treatment and control group and both conversion and revenue/spend as target variables. All our models focus on modeling the treatment effect as additional revenue gained by the treatment, thus focusing on the continuous spending variable as the target.
The binary conversion variable is used for stratification of the data.

All datasets show an inherent treatment effect, i.e. the spending of customers in the treatment group is higher than spending of customers in the control group. This treatment effect is what our analysis tries to isolate and assign to single individuals/sessions in our data.

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/ate-plot.png">

All datasets show an imbalance in the spending class, meaning that the cases where the spending is >0 are quite rare. This is an important caveat which might influence model learning, since the cases where people actually spend money are both the most interesting and the most rare.

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/spend_percentage.png">

<br>
#### Kevin Hillstrom Data
The first dataset was the well known Hillström dataset published by Kevin Hillström in 2008. It contains well cleaned data which revolves around an email campaign that was sent out to a group of male and female customers of an online shop.

As taken from the Blog description, the dataset contains 64,000 customers who last purchased within twelve months. The customers were involved in an e-mail test.

* 1/3 were randomly chosen to receive an e-mail campaign featuring Mens merchandise.
* 1/3 were randomly chosen to receive an e-mail campaign featuring Womens merchandise.
* 1/3 were randomly chosen to not receive an e-mail campaign.

In our case, we do *not* differenciate between the two treatments. Instead, we try to optimize the targeting decision within the population supplied, regardless of mens or womens treatment.

During a period of two weeks following the e-mail campaign, results were tracked. Hillstrom suggests that the user tries to find out whether the mens or the womens campaign was more successful.

The data and its full description can be found here: [https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html]

The Code for reproduction of the results can be found in our GitHub Repo: 
[https://github.com/lukekolbe/hu_apa/tree/master/R%20Code/hillstrom]

<br>
#### Fashion A, Fashion B, Books&Toys
The chair of Information Systems at the Humboldt University of Berlin provided us with three real-world e-commerce datasets from a cooperating partner. These datasets, namely FashionA, FashionB and Books&Toys contain 1.000.000, 700.000 and 150.000 customer sessions, respectively. The data was collected over a one year period. All datasets are similar in structure, since they all possess the same variables. 

All three sets share the same 93 features, which can be grouped into campaign related variables, time counting variables, session status variables (interaction counter, time variables, the target and treatment variables) [.........]

While the data has the same inherent features, the context of the campaign might be different each time. Also, the actual nature of the campaigns showed to be quite different:

The treatments were defined by their Campaign Unit (Currency or Percent), the Campaign Value (either a fixed discount amount or a discount percentage) and the Campaign Minimum Order Value (MOV). The latter defined whether the user had to reach a certain shopping cart value before becoming eligible for treatment. This is very interesting, as it means that being in the treatment group does not necessarily mean everyone treated gets a discount. Instead, the discount is conditional on reaching the MOV.
For any cost and profit analysis, this is very important.

In **Fashion A**, a user had to reach 125€ before getting to claim the 20€ discount proposed by the campaign. The goal here seems to have been an upselling effect: the high MOV reduces the number of people who become eligible for the discount. For everyone else, the assumption is that showing them the option of a discount motivates them to shop. In many cases, they might find find items they like, but not for more than the the MOV. Yet still they might commit to the cart items they added and check out. In such cases, there might be a positive effect of the campaign on the amount spent, without causing any cost.
Some ~64.000 rows in the Data had a differing campaign: 50€ MOV and a 5€ campaign value. These lines were removed because they had a lower average treatment effect than those in the majority campaign and were vastly outnumbered. 

```R
mean(f_a$checkoutAmount[f_a$controlGroup==0&f_a$campaignValue==500]) - 
mean(f_a$checkoutAmount[f_a$controlGroup==1&f_a$campaignValue==500])
```
*0.250630*
```R
mean(f_a$checkoutAmount[f_a$controlGroup==0&f_a$campaignValue==2000]) - 
mean(f_a$checkoutAmount[f_a$controlGroup==1&f_a$campaignValue==2000])
```
*0.4990753*

This left us with a completely homogeneous campaign structure in Fashion A: 
* Campaign Value of 20€

* Campaign Minimum Order Value of 125€

In **Fashion B**, the treatment was similar, but with a lower coupon value (5€) and lower MOV (50 and 30€).
In **Books&Toys**, there was a mix of fixed discounts and percentage on final cart sum.

In all of these datasets, the target variable _checkoutAmount_ (the final amount paid) has the cost of the treatment already internalized. This is relevant for the calaculation of profits later on.

Because of the differences between the datasets, we opted to treat every dataset individually regarding feature selection and cleaning.

<br>
#### Data Cleaning
Since the Hillstrom Data is in a very clean state as-is, no major transformation of the data was done. Only factor variables were dummified in order to make them compatible with the causalBoosting algorithm. Also, the treatment vector, a fractor with 3 levels (No E-Mail, Womens E-Mail, Mens E-Mail) was transformed into a binary treatment vector.

The cleaning process for Fashion A, Fashion B and Books&Toys followed the same steps, respectively. The primary goal of our thorough feature reduction was computational performance: as we had to train a set of 5 models on each of the datasets, we could not afford to run models for extended periods of time. This is why we chose to narrow down the set of features while retaining the maximum possible information value.

An outlier treatment was not done

<br>
##### Feature creation
Firstly, we transformed the unusual controlGroup variable into a treatment variable, effectlively reversing the scale. This is needed as most of our models rely on a binary treatment vector.

Secondly, we created an "eligibility" variable, indicating whether a person i's checkountAmount plus campaignValue puts them over the Minimum Order Value (MOV):
**(checkoutAmount <sub>i</sub> + campaignValue <sub>i</sub> | treatment <sub>i</sub> = 1) >= campaignMov <sub>i</sub>**

After calculating the eligibility, we computed the ExpectedDiscount, which indicates how much we expect each person's treatment (given eligibility) to cost:

**(ExpectedDiscount <sub>i</sub> | eligibility = 1 & campaignUnit = "CURRENCY") = campaignValue <sub>i</sub>**
**(ExpectedDiscount <sub>i</sub> | eligibility = 1 & campaignUnit = "PERCENT") = (checkoutAmount <sub>i</sub> / (1 - campaignValue)) - checkoutAmount**

<br>
##### 1. Remove unneccessary lines
Some features such as trackerKey, campaignId, campaignTags were not useful for analysis. These features were deleted.
A variable called label was also not needed, since it contained a transformed binary target variable (see Gubela 2017) that was likely indended in order to do machine learning with conventional methods. As we model treatment effects directly, this was not necessary (see above).

Some other deleted features include factors with only one level, variables with only NA values and variables with an NA ratio of more than 85%.

<br>
##### 2. NA treatment
After deleting the features with highest NA ratios, we undertook a median imputation on time variables (as setting time counters such as timeSinceLastVisit to 0 would imply a very recent visit while the opposite is likely the case).

For other counters or binary variables, we set NA values to 0.

<br>
##### 3. Correlated features
In order to further reduce the features, we conducted a correlation analysis and removed features with high correlation. The threshold was set to 0.75. This is rather low, the default value of the used findCorrelation function of the 'caret' package is 0.9. But since we needed to reduce features for the costly models to be deployed, we set the threshold to a lower level.
For all datasets (barring Hillstrom), this removed between 11 and 12 features, leaving about 60 features for possible training plus targets and campaign specific features.

<script src="https://gist.github.com/lukekolbe/497a9c4a6d2851028c06230292825ffd.js"></script>
_Exemplary correlation handling_

<br>
##### Feature Selection: Adjusted Net Information Value (NIV)
To further reduce the amount of features, we deployed the NIV function from Leo Guelmann's 'Uplift' Package [8].

The net information value is derived from the well known Information Value (IV) and optimized for purposed of uplift modeling. It is computed as

$$NIV = 100 \times ∑_{i=1}^{G}(P(x=i|y=1)^{T} \times P(x=i|y=0)^{C} - P(x=i|y=0)^{T} \times P(x=i|y=1)^{C}) \times NWOE_i$$

where $$NWOE_i = WOE_i^{T} - WOE_i^{C}$$.

The adjusted net information value is computed as follows:

1. Take B bootstrap samples and compute the NIV for each variable on each sample
2. Compute the mean of the NIV $$(NIV{mean})$$ and standatd deviation of the NIV $$(NIV{sd})$$ for each variable over all the B bootstraps
3. The adjusted NIV for a given variable is computed by adding a penalty term to the mean NIV: 
$$NIV{mean} - \frac{NIV{sd}}{√{B}}$$

<script src="https://gist.github.com/lukekolbe/9b5a09776fc547b6b235700ac61aa890.js"></script>
From the results of the NIV, the top 25 features were selected for training.

<br>
#### Stratification
Since the datasets at hand were quite different in size, we opted to stratify them in a way that leaves us with training (and test) sets of similar size. The challenge was to find a way to keep the balance of the data: each set should have a ratio of treatment and control group members very close to the initial dataset. At the same time, we wanted to make sure that the ratio of conversions (actually buying something) was also kept similar.

The package 'splitstackshape' allows for such stratification across multiple variables and was used to receive our training and test sets. The ratios by which to split the data depended on the size of the dataset.

<script src="https://gist.github.com/lukekolbe/dd9bf80fdbf74e0f9844d0e6c40a70de.js"></script>
<br>
### Model Selection

<br>
#### Approaches to Uplift modeling 
There are three main approaches in the uplift literature, the Class Transformation approach, the Two-Model approach and the direct modeling approach[7]. 
The Class Transformation method is popular but yields mixed results, while some argue that the performance of this method is typically outperformed by the Two-Model approach [6], others say the exacte opposite[7].
Since, there is no clear opinion on this matter, we decided to use the Two-Model approach as a baseline model, which finds wide application as such.

<br>
#### Two-Model approach
The Two-Model approach, how the name suggests, uses two predictive models, one using the treatment group data and the other using the control group data. This makes it possible to use a variety of machine learning models, which is the biggest advantage of this approach. The prediction of these two models get subtracted from each other to generate the uplift predictions. However, this could be a major drawback, for two reasons. First, since we employ two models, we end up having two error terms and secondly and more important both models focus on predicting the outcome isolated from eachother, instead of focusing on the differences between the two groups, which could lead to missing the "weak" uplift signal [7].

<br>
#### Direct Model approach 
There are various approaches to model uplift directly, for example Logistic Regression, K-Nearest-Neigbors, Support Vector Machines. But the by far most frequently used ones are in the family of tree-based models, such as Decision Trees [10], Random Forest [11], Boosting[12] and Bayesian Additive Regression Trees [13]. According to Gubela et. al tree-based models are among the top performers [14].

<br>
#### Model Selection
As mentioned above, tree-based models are usually outperforming other models. This is why we opted to use those mentioned in the Direct Model approach section along with a Ridge Regression Two-Model approach as our baseline model. 

<br>
#### Honest Causal Tree
Since the Causal Tree is the base learner of most of our utilized models, we will quickly summarize its functionality. 
Causal Trees do not differ much from traditional Decision Trees, the only difference is instead of estimating a mean outcome in each leaf it estimates the average treatment effect.
This is achieved by using a modified Mean Sqaured Error criterion for splitting. At each leaf it partions the data in such way that it maximizes the heterogeneity in treatment effects between groups and penalizes a partition that creates variance in leaf estimates [9]. The heterogenieity is usually measured with the Kullback-Leibler divergence [4].

$$ \begin{equation} \label{eq:kl} \displaystyle \mathrm{KL}( P | Q) = \sum_x P(x) \log \frac{ P(x) }{ Q(x) } \end{equation} $$

But how does the tree make sure, that the heterogeneity is not due to noise in the data. The solution is to make the Causal Tree "**honest**" by splitting our training data into two subsamples, where one is functioning as a subsample to find splits and the other is used to estimate the effects employing the splits found in the splitting subsample [9]. Each case of the estimation subsample is dropped down the tree until it falls into a leaf. This procedure is basically mimicing Cross-Validation. 
Cross-Validation per se cannot be applied in uplift settings, because of the fundamental problem of Causal Inference. For every individual, only one outcome is observed, never both at the same time. 

<br>
#### Honest Causal Forest
The Honest Causal Forest is an just like any other Random Forest, an ensemble of Honest Causal Trees. The idea is the same, instead of fitting a single Honest Causal Tree, which can be noisy we fit multiple Causal Honest Trees at the same time, effectively 
The “Random Forest” part functions the same way as a standard Random Forest (e.g., resampling, considering a subset of predictors, averaging across many trees, etc.)[5][11].

<br>
#### "Cross Validated" Honest Causal Boosting
Causal Boosting is also an ensemble algorithm which uses Causal Trees as a "weak" base learner. This change results in the learned function to be able to find treatment effect heterogeneities [12]. 
This Causal Boosting algorithm is validated the same way as the Causal Tree, but the creators of this algorithm still referred to it as "Cross Validated", although this is rather an "Honest" Causal Boosting algorithm. 

<br>
#### Causal Bayesian Additive Regression Trees (BART)
Causal BART is very similar to Causal Boosting, both use sequential weak learners. But instead of applying a learning rate to the sequentilals, the Bayesian approach uses a set of priors. 
By using a prior, which provide regularization in a way that no single regression tree can dominate the total fit, and likelihood to get a poterior distribution of our predictions[13].

<br>
### Model Deployment and Training
We put these up against the two-model approach which functions as our benchmark model. In this regard we want to find out if these specially constructed algorithms outperform the traditional and rather fast implementations, here the ridge regression two modell approach.

<br>
#### Baseline: Two-Model with Ridge Regression
In order to judge whether the direct modeling of treatment effects is more efficient than the conventional two-model approach, we built such a model as baseline. We picked Ridge Regression since it penalizes on coefficient size without removing coefficients entirely, as LASSO does.

<script src="https://gist.github.com/lukekolbe/317c7c868df54edeab45c0cd376dd9c9.js"></script>
<br>
#### CausalTree
As it is a popular and well established method for treatment effect modeling and the foundation of causal_forest() and causalBoosting() models, we naturally fit a causalTree (package 'causalTree'), a derivative of Leo Breiman's Decision Trees geared toward modeling causal inference.

The tree was configured as giving an 'honest' estimation, i.e. the tree was fit on the training data and the nodes were estimated using the (separate) estimation set. Likewise, splitting criteria and cross-validation parameters are also optimized for honest estimation.

Computation-wise, the model is surprisingly costly. The CP parameter was crucial for model predictive power as well as computation time. Setting it too high and the resulting tree only has one node. Setting it too low and the tree becomes overly complex and costly to compute. The difference between cp=0.0000001 (1e<sup>-7</sup>) and cp=0.00000001 (1e<sup>-8</sup>) amounted to 10-fold increase in computation time

<script src="https://gist.github.com/lukekolbe/27513b368226dcc1ee1096be6421d1c9.js"></script>
<br>
#### CausalForest
The Causal Forest (package 'grf) was fitted using a parallelized approach, building four forests with 1000 trees each and then combining them using the designated combine function of the 'grf' package. Parallelization was deployed using the 'doParallel' package. Performance this way in terms of computation time was good.

The Forests were again configured toward honest estimation. All NULL variables were automatically tuned.

<script src="https://gist.github.com/lukekolbe/be6a0ded7db88dab6b3cce96124d6d33.js"></script>
<br>
#### CausalBoosting
In theory a powerful model, the Causal Boosting (package 'causalLearning') model suffered from severe performance issues in our application. The model in its base configuration took over a week to compute (it was stopped after 7 days) and was thus not feasible for implementation. Instead, we de-tuned it by reducing the number of trees to 50 (default = 500), setting the learning rate (eps) to a rather weak 0.3 and reducing the cross validation folds to 5. Still it took up to 36 hours on designated servers to compute.
It is entirely possible that this low-cost approach has negative influence on performance, but computational cost is a factor to consider when choosing models.

<script src="https://gist.github.com/lukekolbe/2bc00088b843ae785915df5575f674c1.js"></script>
<br>
#### Causal Bart
Implementation of Causal Bart (package 'bartCause') was rather straightforward. Computation was rather quick, even with 1000 trees and elaborated sampling. An interesting feature of the bartc() function is the parameter group.by, which lets the model calculate the treatment effects for different groups. This might be useful when there isn't just one treatment, but in fact different treatmens within a population (see description of Books&Toys dataset).

<script src="https://gist.github.com/lukekolbe/455e31d86fb7ce96ad08993cd7e81cf3.js"></script>
<br>
### Implicatons towards Profit, Revenue and Costs. 

We have two different kind of costs. Cost-type one are costs which result from using a voucher, effectively reducing the revenue a customer generates for the amount of the campaign Value. 
The other cost-type stems from not treating customers, which would have generated an additional revenue, if they were treated. This could be seen as some form of opportunity costs. 

The following definitions should help to understand the complexity of incurring costs:

(1) discount<sub>i</sub>:= Campaign Value<sub>i</sub>; if basketValue<sub>i</sub> >= MOV
                       <br>0; else 

(2) Profit<sub>i</sub> := basketValue<sub>i</sub> – discount<sub>i</sub> := checkoutAmount<sub>i</sub>

One should bare in mind, that the cost-type one, costs from discounts are already internalized in the checkout amount variable (1). The discount can either be equal to the campaign value or zero, follwing the logic of (1). As, stated in (2) the checkout amount is equal to the profit.
    
Hence, we do profit modeling when utilizing the _checkoutAmount_ variable as the target variable in our models. 

(3) basketValue<sub>tr</sub> := basketValuect + Uplift<sub>tr</sub>
The basket value of a treated person is equal to the basket value of a person in the control group, plus the uplift in revenue which was generated by the treatment. 

(4) basketValue<sub>ct</sub> := checkoutAmount<sub>ct</sub>
The basket value of a person in the control group is equal to its checkout amount, since there are no additional factors, such as discount or uplifts, since this person was not treated. 

<br>
### Model Evaluation
Since there is no ground truth to evaluate our estimates on, finding a measure for the model performance in Uplift modeling is a core issue. 

Most practicioners resort to measures such as uplift curves [16]. 
In the following example, we are going to show you how to compute these step-by-step.

We are starting with ranking our model predictions for both treated and control data points, and rank them from high to low. 

<script src="https://gist.github.com/lukekolbe/18eb7c3330a42bb390f6db54c2954372.js"></script>
<br>

We continue binning the individuals accordingly to their rankings into deciles. Consequently the individuals with the highest uplift scores will populate the first group. 

<script src="https://gist.github.com/lukekolbe/bc5c40f0f0ac8d9844daae7f4c045cb1.js"></script>
<br>

Next, we are interested in the actual values, especially the following:

* Number of individuals in the treatment group per decile

* Number of individuals in the control group per decile

* The spending of both groups per decile

* The spending per capita of both groups per decile

* The achieved uplift per decile 

<script src="https://gist.github.com/lukekolbe/a2df2f2cc1c15f799f5ac24cd8804a0b.js"></script>
<br>

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/perf_pred_b_t_bc.png">

<br>Based on these values we can create some plots, visualizing the performance matrix. 
Ideally, the plot should have high average checkout amount for the treatment group and low average checkout amounts for the control group in the first deciles and vice versa in the last deciles. 

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/BooksToys_causalBart_arpd_.png">

By subtracting the average checkout amount for the control group from the average checkout amount of the treatment group we achieve the uplift per capita per decile as pictured in the following plot.

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/BooksToys_causalBart_upd_.png">

Now using our beforehand calculated performance matrix we are able to generate uplift curves. The procedure is as follows:

1.	We calculate the cumulitative sum of the revenue of the treated and the control group while controlling for the total population in each group of the specified decile.

2.	We calculate the percentage of the total amount of people (both treatment and control) present in each decile. 

<script src="https://gist.github.com/lukekolbe/42c33b759d727fac25ebbcfe9f7bad1e.js"></script>
<br>

Afterwards, we are able to calculate the incremental gains for the model performance. This represents our uplift curve. 
The overall incremental gains represent the random curve. This is corresponding to the global effect of the treatment.

<script src="https://gist.github.com/lukekolbe/c3324c449bb15d33ba06e464ff89207f.js"></script>
<br>

In our case the positive slope of the random curve confirms the overall positive effect of the randomized campaign. Finally, the shape of the curves shows the strong positive and negative uplifts. Contrary, the closer an uplift curve is to the random line, the weaker are the uplifts achieved. 


We proceed with computing the area of both of these curves. 
<script src="https://gist.github.com/lukekolbe/490216b804451790f4a88dfa3b2a6caa.js"></script>
<br>

Next up we substract the area under to random curve from the area of our uplift curve. 

<script src="https://gist.github.com/lukekolbe/c6d024d8c1a880e76d13b51157b4e865.js"></script>
<br>

The area between the Qini curve and the Random curve is called the Qini-coefficient.

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/BooksToys_causalBart_qini_.png">

The code was adapted from Guelman's Uplift package. 
https://cran.r-project.org/web/packages/uplift/index.html

<br>
### Model Profitability

Since we now want to know how profitiable a campaign would be which is based on our model predictions, we need to know the revenue and cost streams which would occur and compare these to the realized campaign from the random experiment. Thus, we need to calculate the additional revenue, cost, and finally profit we generate.

From here on we refer to the campaign based on the model as the _model campaign_ and for the realized camapign from the random experiement as the _campaign_.
    
**Assumption**
We assume, that every individual who is eligible to use a voucher, indeed uses it. This is rather a conservative point of view. But we have no indication of how many people actually used the discount code for every dataset.  

In the first step we need these variables, two of these, namingly _eligibility_ and _ExpectedDiscount_ should be familiar to this point:
    
* eligibility
* model eligibility
* ExpecedDiscount
* model ExpectedDiscount

So what are the two other variables referring to our model?
    
The _model eligibility_ ("m.eligiblity") variable is an adaption of the _eligibility_ variable we spoke about in our **Feature Creation** section. 
The difference is that _model eligibility_ is applied to all individuals, internalizes the uplift (treatment effect), and is not accounting for the campaign value (discount). The reason for the lastest is, that you cannot use a voucher before qualifying to use it. Effectively, in this case the checkout amount is equal to the revenue and is not internalizing the discount. 

This function demonstrates how the _model eligibility_ variable is constructed: 

**(p + checkoutAmount <sub>i</sub>) >= campaignMov <sub>i</sub>**

After calculating the _model eligibility_, we computed the _model ExpectedDiscount_, which indicates how much we expect each person's treatment (given _model eligibility_) to cost.
Notice that individuals receiving a discount in percent are calculated slightly different than those with an absolut discount value:

**(ExpectedDiscount <sub>i</sub> | eligibility = 1 & campaignUnit = "CURRENCY") = campaignValue <sub>i</sub>**
**(ExpectedDiscount <sub>i</sub> | eligibility = 1 & campaignUnit = "PERCENT") = (checkoutAmount <sub>i</sub> / (1 - campaignValue)) - checkoutAmount**
**Costs**

These new variables are added to our existing model matrix. 
    
<script src="https://gist.github.com/AsmirMumin/d178be6d9d00d824e834d046bc870194.js"></script>
We proceed with calculating the number of _eligible_ and _model eligible_ customers per decile to be able to put these in our profitability matrix later on.
The difference of these measures is the number of additional eligible customers, which we called _delta.eligible_.

<script src="https://gist.github.com/AsmirMumin/29de48f8a851911f8494080c4b4fe522.js"></script>
The number of aditionally eligible individuals is core to compute the additional costs stemming from our _model campaign_, since only these individuals can use the discount voucher, which effectively reduces our profits and thus adds costs.
The model costs are generated by individuals using their vouchers multiplied by the campaign value (discount) of these vocuhers. Same applies for the campaign costs of the _campaign_.
The difference of these costs are the additional costs caused by our _model campaign_, and we are only interested in these. 
    
<script src="https://gist.github.com/AsmirMumin/73edf863c8b93efae170d77ee2dd29ad.js"></script>
<br>

The additional revenue is basically the number of additionally treated individuals multiplied by the uplift achieved on a decile level. 
Now, we are able to calculate the additional profit, which is generated by our _model campaign_ if we treated every customer for each decile. 
    
<script src="https://gist.github.com/AsmirMumin/f0a08e6a42fb2d6f15f10443e1bbab28.js"></script>
<br>

But since it is not optimal to treat everyone. We need to compute the additional profits generated cumulatively for each treated decile to detect how many deciles we should treat.  
This is done, by cumulating profits for each treated decile and accounting for the missed profits of deciles which are not treated. We did this for every possible number of deciles and finally found the maximum. This is telling us how much profit we gain and at how many deciles we should treat. 
    
<script src="https://gist.github.com/AsmirMumin/e58e83c4fd9e00f4f71691634bb0a3ee.js"></script>
<br>

Since, it is not always possible to treat as many as suggested when maximizing the profit, due to budget constraints. We opted to calculate the Return on Investment (ROI) as well and use it as our decision criterion. This enables marketers to gain the maximal profitability per euro spend.
We save these metrics to a profitability matrix. 

<script src="https://gist.github.com/AsmirMumin/16706b352d3fa586926ce0a457a2890d.js"></script>
<br>

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/profitability_b_t_bc.png">

<br>
### Results
So how did our more sophisticated models, which are directly measuring the uplift in comparison to our baseline model, the Two-Model approach? 

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/Qini_table.png">
<br>

The tree based models dominated the Two-Model approach measured by the Qini-coefficient in our FashionA and FashionB dataset. In the BooksToys dataset, the Two-Model approach placed second to Causal BART and in the Hillstrom dataset it placed best, leaving out the ensemble model. 
    
We could not confirm that having more features generally betters the model performance. 

Which models are especially mention-worthy?
The Causal Bart generally performed pretty well across all datasets, but FashionB. It was only beaten by our ensemble model, which averages the predictions of the other models did well in every dataset, and is considered the overall winner when measuring the Qini-coefficient. 

Although the Qini-coefficient is a viable measure for the model performance, it does not account for costs which are occuring from the treatment. Thus, these results are only fully valid, when we have low or no costs from additional treatments. 
Since costs are occuring in some of the datasets, the of models can change. 

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/Profit-decile_table.png">
<br>

Considering the absolute profits gained by the models, the ensemble model no longer is the best performer. 
We also do not have a clear winner here. In the FashionA dataset, the Causal Forest is suprisingly the winner, with 13k profits generated when targeting 9 deciles. In FashionB causal Boosting is generating the most profits, with 17k while treating 8 deciles. In BooysToys we achieved the highest profit gain of 50k, when treating only the first decile, which makes it also the most profitable one in terms of Return on Investment (ROI). 
In the Hillstrom dataset, Causal Boosting won again with 4k profits while treating all customers. 
    
Causal Boosting was able top place first in two of our four datasets, making it the best performing in terms of absolute profits. 
    
If budget is restricted, the ROI would be the measure to decide on which deciles to treat. 

<img style=" width:750px;display:block;margin:0 auto;"src="https://raw.githubusercontent.com/lukekolbe/hu_apa/master/final%20predictions/final%20plots/ROI_table.png">
<br>

You might ask yourself, why are one of the latest deciles the most profitible ones, this is due to the campaign design. We explain this in the next section.  

<br>
### Key Findings
We see that the way the campaign is constructed/designed is heavily impactful on the uplifts achieved and the general profitability of our campaign.
For example, in FashionA, the marketers have used upselling, a technique employed in marketing to incentivize customers to spend a certain amount of money. In this case, the treatment group received a 20€ voucher, but those vouchers could be only used if the customers had a minimum order value of 125€. 
The data shows that not everybody was willing to spend this amount of money, since a lot of treated customers didn't reach the necessary spending, but still we estimated a lot of customers to respond positively on the treatment. This means, that solely giving the customers a chance of saving 20€ is effective as a treatment, which means it incentivizes customers to shop. The data suggests that some customers would browse the shopping website for products they like, finally ending up at a lower cart value than the minimum order value requires, and still buy those articles. One could assume that they were invested too much to this point, be it timewise or emotionally to churn now. 
The profit analysis for FashionA shows that the most profitable deciles for every model were always the last ones, which indicates that this upselling technique allows to treat most of the customers, since the cost which come with the 20€ voucher would only occur for a minority of the treated. 
The same logic applies to dataset FashionB, where the minimum order value is between 30€ and 50€.
Similarly, for the Hillstrom data, where treatment is generally very cheap, it might still be benefitial to treat many people – only avoiding the sleeping dogs (customers who have negative treatment effects).

Contrarily, if we don't restrict the treatment, every additional treatment equates to additional cost. The profitability table above shows that in Books&Toys, the lower deciles are most profitable. This is likely because many rows in the data have no MOV assigned, meaning that everyone who is treated receives a discount. In these cases the treatment cost amounts to substantial amounts when treating all or most people. Hence, the profit as we model it does not only reflect the uplift gained, but also the money _not_ spend by not treating later deciles.

<br>
### Conclusion

In Conclusion, we were able to increase our model profits with every model. 
Our results suggest that uplift modeling is a viable tool to optimize marketing campaign profits. 
But one should bare in mind, the cost of treatments if the number of the effectively treated is high or has no restriction. 
The campaigns with a higher minimum order value where most profitable. 

Our models mostly performed better than random. But it is hard to declare a clear winner, since it is highly depended what our campaign looks like. 

<br>
### Recommendations
We recommend to use various models to find the best performing ones for a given scenario, regarding budget restrictions and campaign design.

It is also important to account for the computational costs: Causal Boosting, which was computationally very heavy, might not be a sensible choice for treatment effect modeling when time is a factor or model adaption and re-training cycles are frequent.

Deciding for a model should ideally consider both the performance as measures by the Qini score, as well as the (predicted) profitability. Builing ensembles might be useful!

Also regarding the modeling, it is much simpler to assess the profitability when you are not dealing with a variable which has internalized costs. This proved to be quite troubling when finding the right assessment models, as for some lines the checkoutAmount is acutal revenue, while for others it is revenue minus treatment cost.

<br>
### Literature
[1] M. Soltys and S. Jaroszewicz. Boosting algorithms for uplift modeling, 2018.
    

[2] B. Hansotia and B. Rukstales. Incremental value modeling. Journal of Interactive Marketing, 16(3):35–46, 2002.
    
[3] K. Hillstrom. The MineThatData e-mail analytics and data mining challenge. MineThatData blog, http://blog.minethatdata.com/2008/03/ minethatdata-e-mail-analytics-and-data.html, 2008. Retrieved on 06.10.2014.

[4] I. Csiszar and P. Shields. Information theory and statistics: A tutorial. Foundations and Trends in Communications and Information Theory, 1(4):417–528, 2004.
    
[5] L. Guelman, M. Guill'en, and A.M. P'erez-Mar'ın. Random forests for uplift modeling: An insurance customer retention case. In Modeling and Simulation in Engineering, Economics and Management, volume 115 of Lecture Notes in Business Information Processing (LNBIP), pages 123–133. Springer, 2012.

[6] M. Ja'skowski and S. Jaroszewicz. Uplift modeling for clinical trial data. In ICML 2012 Workshop on Machine Learning for Clinical Data Analysis, Edinburgh, Scotland, June 2012.

[7] P. Gutierrez and J.-Y. Gérardy. Causal Inference and Uplift Modeling. A review of the literature, 2016.

[8] L. Guelman, M. Guillen, A. M. Pérez-Marín, et al. Optimal personalized treat-
ment rules for marketing interventions: A review of methods, a new proposal, and an insurance case study. Technical report, 2014.

[9] S. Athey and G. W. Imbens. Recursive partitioning for heterogeneous causal effects.
arXiv preprint arXiv:1504.01132, 2015a.

[10] S. Athey and G. W. Imbens. Machine learning methods for estimating heterogeneous causal effects. stat, 1050:5, 2015b.

[11] S. Athey and S. Wager. Estimation and inference of heterogeneous treatment effects using random forests. arXiv preprint arXiv:1510.04342, 2015.

[12] S. Powers et. al. Some methods for heterogenous treatment effect estimation in high dimensions. arXiv preprint arXiv:1707.00102, 2017.

[13] P. R. Hahn et. al. Bayesian Regresion tree models for causal inference: Regularization, Confounding, and Heterogenous Effects. arXiv preprint arXiv:1706.09523, 2019.

[14] Gubela et. al. Revenue Uplift Modeling, 2007. 

[15] J. L. Hill. Bayesian Nonparametric Modeling for Causal Inference, Journal of Computational and Graphical Statistics, 20:1, 217-240. DOI: 10.1198/jcgs.2010.08162, 2011.

[16] M. Soltys, S. Jaroszewicz, and P. Rzepakowski. Ensemble methods for uplift modeling. Data mining and knowledge discovery, 29(6):1531–1559, 2015.

[17] https://www.statista.com/statistics/236943/global-advertising-spending/

