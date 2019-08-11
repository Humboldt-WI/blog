+++
title = "APA19 Microfinance Policy"
date = '2019-08-15'
tags = [ "Causal Inference", "Class19",]
categories = ["Course projects"]
banner = "img/seminar/nn_fundamentals/neuralNetworkKlein.jpg"
author = "Seminar Applied Predictive Modeling (SS19)"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = "One sentence description of your work"
+++

# Microfinance Policy

#### Authors: Edanur Kahvecioglu and Yu-Tang Wu

## Abstract

Is microcredit a miracle or just a hype? While more and more researches were conducted to study microcredits, people started to question the effectiveness of microcredit program.  This project uses the experiment data of Banerjee et al. (2014) and explores the heterogeneity of easy access to microcredit to households from different areas in Hyderabad, India. Causual Random Forest and Two-model approach are used to analyze treatment effects on the household level and identify the important variables that separate households with higher treatment effect.

### Table of Contents:

1. [Motivation](#motivation)
2. [Introduction](#introduction)
   1. [Why Index Variables](#why-index-variables)
3. [Dataset](#dataset)
4. [Methods](#methods)
   1. [Structure of Analysis](#structure-of-analysis)
   2. [Two-model Approach And GATES](#two-model-approach-and-gates)
   3. [Causal Random Forest](#causal-random-forest)
5. [Results](#results)
   1. [Business Index](#business-index)
   2. [Women Empowerment Index](#women-empowerment-index)
   3. [Other Indexes](#other-indexes)
6. [Model Comparison](#model-comparison)
7. [Conclusions](#conclusions)
8. [References](#references)

---

## Motivation

Microfinance, is a type of credit provided to unemployed or low-income
individuals or groups who are deprived of having access to
conventional financial services. Although it is not a new concept in
human history, as a first organization, Grameen Bank was established
in 1976 by Muhammad Yunus in Bangladesh. It provided small loans to
the poor without requiring collateral. 

## Introduction

The paper ([Banerjee
et al. 2015](#banerjee2015MiracleMicrofinanceEvidence)) presents the
results from the randomized evaluation of a group-lending microcredit
program in Hyderabad, India. Microcredit has been lent to groups of 6
to 10 women by a microfinance institution, Spandana. According to the
baseline survey conducted in 2005, 104 neighborhoods with no
pre-existing microfinance was selected. They were paired up and one of 
each pair was randomly assigned to the treatment group. Spandana then
progressively began operating in the 52 treatment areas between 2006
and 2007. As an important point to highlight, the treatment was not a
credit take-up but an easy access to microcredit.

### Why Index Variables

Instead of analyzing specific household behaviors such as business
investment or consumption on nondurable goods, as the original paper
did, we want to focus on the more general aspect of the
well-being. This could be done by utilizing the “Index” variables
provided by the dataset.

These index variables are the weighted average of related
variables. For example, business index is the weighted average of
business asset, business expenses, business profit etc. By doing so,
the index variable become a holistic representation of an aspect of
well-being and can capture general condition of that aspect. In the
example above, business index could represent the overall
circumstances of the business activities of the households.

## Dataset

The paper ([Banerjee et
al. 2015](#banerjee2015MiracleMicrofinanceEvidence)) comes with 5
datasets, including data gathered in the first endline survey and
second endline survey as well as data gathered in the baseline
survey.

This project uses mainly the endline dataset, specifically the first
endline data. The baseline dataset would be useful to calculate the
changes across time. However, in our case, the index of households is
different between baseline dataset and endline dataset. Hence we can
not make one-to-one mapping between households in the two
datasets. And since we are mainly interested in how easy accessibility
to Spandana microcredits affect households well-being, only the data
collected in the first endline survey will be used in this project. 

!The helper functions we used in this project can be found
[here](https://github.com/thmstang/apa19).

``` r
endline1 <- load_endline1()
str(endline1)
```

    ## 'data.frame':    6129 obs. of  56 variables:
    ##  $ hhid                      : int  1 2 3 4 5 6 7 8 9 11 ...
    ##  $ areaid                    : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ treatment                 : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ old_biz                   : num  0 0 1 1 1 1 0 0 1 0 ...
    ##  $ hhsize_adj_1              : num  2.8 3.24 4.18 4.03 5.41 ...
    ##  $ adults_1                  : num  3 2 2 2 4 3 4 2 7 4 ...
    ##  $ children_1                : num  0 2 3 3 2 3 0 2 0 0 ...
    ##  $ male_head_1               : int  1 1 1 1 1 1 1 1 1 0 ...
    ##  $ head_age_1                : int  20 34 40 37 32 40 43 31 62 48 ...
    ##  $ head_noeduc_1             : num  1 0 0 0 0 1 1 0 0 1 ...
    ##  $ women1845_1               : num  2 1 1 1 1 1 2 1 2 1 ...
    ##  $ anychild1318_1            : num  0 0 1 1 1 1 1 0 1 0 ...
    ##  $ spouse_works_wage_1       : int  0 1 0 1 0 0 0 0 0 0 ...
    ##  $ ownland_hyderabad_1       : num  0 0 0 0 0 0 1 0 0 0 ...
    ##  $ ownland_village_1         : int  0 0 1 0 1 0 0 0 0 0 ...
    ##  $ spandana_1                : int  1 0 0 0 0 1 0 1 0 0 ...
    ##  $ othermfi_1                : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ anybank_1                 : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ anyinformal_1             : int  1 0 1 1 0 1 0 0 0 1 ...
    ##  $ anyloan_1                 : num  1 0 0 1 1 1 1 1 1 0 ...
    ##  $ everlate_1                : int  1 0 1 1 0 0 0 1 0 0 ...
    ##  $ mfi_loan_cycles_1         : num  1 0 0 0 0 3 0 1 0 0 ...
    ##  $ spandana_amt_1            : num  1961 0 0 0 0 ...
    ##  $ othermfi_amt_1            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ bank_amt_1                : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ informal_amt_1            : num  10193 0 6538 6538 0 ...
    ##  $ anyloan_amt_1             : num  12617 0 0 5634 2506 ...
    ##  $ bizassets_1               : num  0 0 2000 0 31700 0 0 0 0 0 ...
    ##  $ bizinvestment_1           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ bizrev_1                  : num  0 0 1800 5000 12400 7560 2170 0 2300 0 ...
    ##  $ bizexpense_1              : num  0 0 205 205 8750 ...
    ##  $ bizprofit_1               : num  0 0 1595 4795 3650 ...
    ##  $ bizemployees_1            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ total_biz_1               : num  0 0 1 1 1 1 2 0 2 0 ...
    ##  $ newbiz_1                  : num  0 0 0 0 0 0 1 0 0 0 ...
    ##  $ female_biz_1              : num  0 0 0 0 0 0 1 0 1 0 ...
    ##  $ female_biz_new_1          : num  0 0 0 0 0 0 1 0 0 0 ...
    ##  $ wages_nonbiz_1            : int  2000 3900 5000 1500 0 500 0 2000 3950 3600 ...
    ##  $ hours_week_biz_1          : num  0 0 21 77 70 126 0 0 152 0 ...
    ##  $ hours_week_outside_1      : num  48 8 0 0 0 0 0 0 0 4 ...
    ##  $ hours_headspouse_outside_1: num  48 4 0 63 0 0 0 32 0 0 ...
    ##  $ hours_headspouse_biz_1    : num  0 4 49 14 70 70 0 0 56 4 ...
    ##  $ durables_exp_mo_pc_1      : num  2.868 0.981 5.54 4.169 2.868 ...
    ##  $ nondurable_exp_mo_pc_1    : num  0.312 0.107 0.604 0.454 0.312 ...
    ##  $ food_exp_mo_pc_1          : num  42.2 73.2 65.4 68.1 44.8 ...
    ##  $ health_exp_mo_pc_1        : num  9.73 6.17 5.21 0 4.03 ...
    ##  $ temptation_exp_mo_pc_1    : num  6.62 0 2.61 0 0 ...
    ##  $ festival_exp_mo_pc_1      : num  1.95 9.25 4.34 9.01 1.34 ...
    ##  $ home_durable_index_1      : num  2.69 2.2 2.46 1.3 2.65 ...
    ##  $ women_emp_index_1         : num  -0.4154 0.5629 -0.0623 -0.368 -0.3653 ...
    ##  $ credit_index_1            : num  1.136 -0.492 -0.417 -0.178 -0.269 ...
    ##  $ biz_index_all_1           : num  -0.224 -0.224 0.0651 0.0827 0.3603 ...
    ##  $ income_index_1            : num  -0.160999 0.081633 0.296675 -0.000669 -0.245753 ...
    ##  $ labor_index_1             : num  -0.1042 -0.6321 -0.3271 0.051 -0.0195 ...
    ##  $ consumption_index_1       : num  -0.1662 -0.0296 -0.088 -0.2396 -0.2112 ...
    ##  $ social_index_1            : num  -0.0783 0.2015 -0.0965 -0.1182 -0.1703 ...
    ##  - attr(*, "na.action")= 'omit' Named int  27 109 185 239 241 290 419 511 529 612 ...
    ##   ..- attr(*, "names")= chr  "27" "109" "185" "239" ...

In total 6129 households completed the first endline survey. There are
56 variables in the dataset which contain the information of a
household’s basic properties (the household size `hhsize_adj_1`, how
old is the household head `head_age_1`), their financial/loans status
(whether they took loans `anyloan_1`, how many bank loan they took
`bank_amt_1`), their businesses status if any (profits `bizprofit_1`,
assets `bizassets_1`), their monthly expenditures on various types of
good (expenditure on nondurable goods per capita
`nondurable_exp_mo_pc_1`), and the calculated index variables.

## Methods

In this section we will present the structure of our analysis as well
as the methods we used this project. 

Traditional Response Models vs. Uplift Modeling (I read about this at
the reference with *, I also write it at the motivation part as one of
the reasons why we picked an uplift model) 

We picked 3 models:

- Causal forest
- Causal KNN (??)
- Two Model Approach

### Structure of Analysis

This section describes the general structure of the analysis and the
methods used for each task. The details of each method will be
discussed in the next section. 

1. Detect whether the treatment effect on the selected index variable exhibits heterogeneity.

    This is vital as our "welfare-enhancing policies" is based on the
    assumption of targeting only the households with relatively large
    treatment effect. Hence if the treatment effect is homogeneous,
    which means all households exhibit similar treatment effect, then
    the targeting policies would be inefficient.  To perform this
    task, Two-model Approach with Sorted Groups Average Treatment
    Effect (GATES), which was proposed by Chernozhukov ([Chernozhukov
    et al. 2018)](#chernozhukov2018GenericMachineLearning)), will be
    used. 

2. Find the variables that separate households with higher treatment effect.

    If the treatment effect is heterogeneous, next thing to do is to
    try to separate those high treatment effect households.  For this
    task we will use to methods, Causal Random Forest and Two-model
    Approach with Random Forest. 

    1. Causal Forest: The Signiﬁcance-Based Splitting Criterion

        The causal forest’s splitting rule builds trees and mimics the
        process that would be used if the treatment eﬀects were
        observed ([Hitsch & Misra
        2008](#hitsch2018HeterogeneousTreatmentEffects)). 

    2. Two-Model Approach

        There is no conventional way to get variable importance using
        Two-Model Approach. However, given the predicted conditional
        treatment effect we could further fit a Random Forest model to
        get the variable importance. 

3. Find the thresholds for given variables that could make the policies more performing.

    Narrowing the target could make the policy more efficient and
    better performing. Hence after knowing which variables make the
    highest impact on households’ treatment effect, our next step is
    then to investigate at which values of those variables a household
    could benefit the most from the treatment.  To accomplish this, we
    will use partial dependence plots to analyze those critical
    values. 

### Two-model Approach And GATES

Two-model approach was proposed by Chernozhukov et al. as a method for
estimating average treatment effects ([Chernozhukov et
al. 2016](#chernozhukov2016DoubleDebiasedMachine), [Chernozhukov et
al. 2017](#chernozhukov2017DoubleDebiasedNeyman)). The concept is
fairly straightforward. Since machine learning has been shown to be
highly effective in prediction, two-model approach exploits this feature
by fitting one machine learning method on the control group and the
other on the treatment group. This way we obtain one model that could be
used to predict the “baseline” effect, which is defined as the outcomes
if the subjects were not treated, and the other model that could be used
to predict the outcomes if they were treated. We could use the two
models to predict baseline effect and treatment effect for each subject.
Then the conditional treatment is simply the difference between them.

Sorted group average treatment effect was proposed by Chernozhukov et
al. as one of the strategies to estimate and make inferences on key
features of heterogeneous treatment effects. ([Chernozhukov et al
2018](#chernozhukov2018GenericMachineLearning)).

### Causal Random Forest

## Results

### Business Index

The first index variable we choose is business index for the reason
that we believe boosting business activities is a more direct way to
stimulate economic growth.  

Studies suggested that small businesses are significant contributors
to economic development, job creation, and general welfare ([Morrison
et al. 2003](#morrison2003SmallBusinessGrowth)) meanwhile
entrepreneurial activity plays an important role in economic growth
([Stel et al. 2005](#stel2005EffectEntrepreneurialActivity)). In
addition, according to Spandana’s initial goal of the experiment, the
microcredit program was aimed to provide a better environment for
women in Hyderabad to create their own small businesses.  


1. Heterogeneity of Treament Effect on Business Index

    ![GATES Business Index](/blog/img/seminar/apa19-microfinance/biz_gates.png)

2. The Groups Separating Variables

3. The Critical Values of Selected Variables

### Women Empowerment Index

1. Heterogeneity of Treament Effect on Business Index

    ![GATES Business Index](/blog/img/seminar/apa19-microfinance/biz_gates.png)

2. The Groups Separating Variables

3. The Critical Values of Selected Variables

### Other Indexes

1. Heterogeneity of Treament Effect on Business Index

    ![GATES Business Index](/blog/img/seminar/apa19-microfinance/biz_gates.png)

2. The Groups Separating Variables

3. The Critical Values of Selected Variables

## Model Comparison

Measuring and comparing the performance of our two methods is not
straightforward as calculating MSE (mean square error) or determining
the accuracy. Since in real life causal inference problems, we do not
have the true treatment effect. Then the above mentioned metrics can
not apply under this circumstance as they need to compare the
predicted value with the true value.  

Usually for uplift models, the metric used to compare the performances
of different methods is Qini score (add references). (maybe add some
advantages of Qini score and why people usually use it).  

However, considering the target variables we selected in this project
are continuous instead of categorical, Qini score is not suitable in
our case. There are ways to circumvent this issue. For example we
could convert the numeric values into binaries based on some
predefined thresholds that indicate whether or not a household
“really” benefit from the treatment. (Add the reasons why we didn’t do
this.)  

The other comparing method is to use the transformed outcomes as
“true” value and calculate the distance of predicted value and the
transformed outcome. This way the conventional error measurement, such
as mean absolute error (MAE) and mean square error (MSE), could be
implemented to assess the model performance. For this project we
decided to use transformed outcome approach with mean square error as
the error metric when comparing the performances of the two methods.  

## Conclusions

## References

<ul>
<li id="banerjee2015MiracleMicrofinanceEvidence">
Banerjee, Abhijit, Esther Duflo, Rachel Glennerster, and Cynthia Kinnan. 2015. <i>The Miracle of Microfinance? Evidence from a Randomized Evaluation.</i> American Economic Journal: Applied Economics, 7 (1): 22-53. 
<br/>
<a target="_blank"
href="https://www.aeaweb.org/articles?id=10.1257/app.20130533">
https://www.aeaweb.org/articles?id=10.1257/app.20130533
</a>
</li>
<li id="chernozhukov2016DoubleDebiasedMachine">
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
& Newey, W. K. (2016). <i>Double machine learning for treatment and
causal parameters</i> (No. CWP49/16). cemmap working paper.
<br/>
<a target="_blank"
href="http://arxiv.org/abs/1608.00060">
http://arxiv.org/abs/1608.00060
</a>
</li>
<li id="chernozhukov2017DoubleDebiasedNeyman">
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
& Newey, W. (2017). <i>Double/debiased/neyman machine learning of
treatment effects.</i> American Economic Review, 107(5), 261-65. 
<br/>
<a target="_blank"
href="https://www.aeaweb.org/articles?id=10.1257/aer.p20171038">
https://www.aeaweb.org/articles?id=10.1257/aer.p20171038
</a>
</li>
<li id="chernozhukov2018GenericMachineLearning">
Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val,
I. (2018). <i>Generic machine learning inference on heterogenous
treatment effects in randomized experiments</i> (No. w24678). National
Bureau of Economic Research.
<br/>
<a target="_blank"
href="https://www.nber.org/papers/w24678">
https://www.nber.org/papers/w24678
</a>
</li> 
<li id="hitsch2018HeterogeneousTreatmentEffects">
Hitsch, Guenter J. and Misra, Sanjog, <i>Heterogeneous Treatment Effects and Optimal Targeting Policy Evaluation</i> (January 28, 2018). 
<br/>
<a target="_blank"
href="https://ssrn.com/abstract=3111957">
https://ssrn.com/abstract=3111957
</a>
</li>
<li id="jacobCausalInferenceUsing">
Jacob, Daniel. (2018). <i>Causal Inference using Machine Learning.</i>
<br/>
<a target="_blank"
href="https://www.researchgate.net/publication/328686789_Causal_Inference_using_Machine_Learning">
https://www.researchgate.net/publication/328686789_Causal_Inference_using_Machine_Learning
</a>
</li>
<li id="morrison2003SmallBusinessGrowth">
Morrison, A., Breen, J., & Ali, S. (2003). <i>Small business growth:
intention, ability, and opportunity.</i> Journal of small business
management, 41(4), 417-425. 
<br/>
<a target="_blank"
href="https://onlinelibrary.wiley.com/doi/abs/10.1111/1540-627X.00092">
https://onlinelibrary.wiley.com/doi/abs/10.1111/1540-627X.00092
</a>
</li>
<li id="stel2005EffectEntrepreneurialActivity">
Van Stel, A., Carree, M., & Thurik, R. (2005). <i>The effect of
entrepreneurial activity on national economic growth.</i> Small
business economics, 24(3), 311-321. 
<br/>
<a target="_blank"
href="https://link.springer.com/article/10.1007/s11187-005-1996-6">
https://link.springer.com/article/10.1007/s11187-005-1996-6
</a>
</li>
</ul>
