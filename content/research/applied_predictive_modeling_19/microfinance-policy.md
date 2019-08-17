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

Is microcredit a miracle or just a hype? While more and more researches were conducted to study microcredits, people started to question the effectiveness of microcredit program. This project explores the heterogeneity of treatment effect on microcredit to households' well-being. Causuall Random Forest and Two-model approach are used to analyze treatment effects on the household level and identify the important variables that separate households with higher treatment effect. The results show evidence of heterogeneity of the conditional treatment effect on certain aspects of well-being. In addition, policy suggestions are proposed to attempt to increase the effectiveness of microfinance.

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
   2. [Business Index Without Expenditure Variables](#business-index-without-expenditures-variables)
   2. [Women Empowerment Index](#women-empowerment-index)
   3. [Other Indexes](#other-indexes)
6. [Model Comparison](#model-comparison)
7. [Conclusions](#conclusions)
8. [References](#references)

---

## Motivation

Microfinance is a type of credit provided to unemployed or low-income individuals or groups who are deprived of having access to conventional financial services. Although it is not a new concept in human history, as a first organization, Grameen Bank was established in 1976 by Muhammad Yunus in Bangladesh. It provided small loans to the poor without requiring collateral.

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
investment or consumption of nondurable goods, as the original paper
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
endline data, for analysis. The baseline dataset would be useful to
calculate the changes across time. However, in our case, the index of
households is different between baseline dataset and endline
dataset. Hence we can not make one-to-one mapping between households
in the two datasets. And since we are mainly interested in how easy
accessibility to Spandana microcredits affect households’ well-being,
only the data collected in the first endline survey will be used in
this project. 

!The helper functions we used in this project can be found
[here](https://github.com/thmstang/apa19).

``` r
endline1 <- load_endline1()
str(endline1)
```

In total of 6129 households completed the first endline survey. There are 56 variables in the dataset which contains the information of a household’s basic properties (the household size `hhsize_adj_1`, how old is the household head `head_age_1`), their financial/loans status (whether they took loans `anyloan_1`, how many bank loan they took `bank_amt_1`), their businesses status if any (profits `bizprofit_1`, assets `bizassets_1`), their monthly expenditures on various types of good (expenditure on nondurable goods per capita `nondurable_exp_mo_pc_1`), and the calculated index variables.

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

1. Detect whether the treatment effect on the selected index variable
   exhibits heterogeneity. 

    This is vital as our "welfare-enhancing policies" is based on the
    assumption of targeting only the households with relatively large
    treatment effect. Hence if the treatment effect is homogeneous,
    which means all households exhibit similar treatment effect, then
    the targeting policies would be inefficient.  To perform this
    task, Two-model Approach with Sorted Groups Average Treatment
    Effect (GATES), which was proposed by Chernozhukov ([Chernozhukov
    et al. 2018)](#chernozhukov2018GenericMachineLearning)), will be
    used. 

2. Find the variables that separate households with higher treatment
   effect. 

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

3. Find the thresholds for given variables that could make the
   policies more performing. 

    Narrowing the target could make the policy more efficient and
    better performing. Hence after knowing which variables make the
    highest impact on households’ treatment effect, our next step is
    then to investigate at which values of those variables a household
    could benefit the most from the treatment.  To accomplish this, we
    will use partial dependence plots to analyze those critical
    values. 

### Two-model Approach And GATES

Two-model approach was proposed by Chernozhukov et al. as a method for
estimating average treatment effects ([Chernozhukov et
al. 2016](#chernozhukov2016DoubleDebiasedMachine),[Chernozhukov et
al. 2017](#chernozhukov2017DoubleDebiasedNeyman)). The concept is
fairly straightforward. Since machine learning has been shown to be
highly effective in prediction, two-model approach exploits this
feature by fitting two independent models, one estimating treatment
effect in the control group and the other in the treatment group. This
way we obtain one model that could be used to predict the “baseline”
outcome, which is defined as the outcomes if the subjects were not
treated, and the other model that could be used to predict the
outcomes if they were treated. Then individual’s conditional treatment
effect is the difference between the predicted baseline outcome and
the predicted treated outcome. 

The main advantages of two-model approach are its simplicity and easy
to implement. It requires no additional methods and is generic to
machine learning algorithms. 

Sorted group average treatment effect was also proposed by
Chernozhukov et al. (2018) as one of the strategies to estimate and
make inferences on key features of heterogeneous treatment
effects. ([Chernozhukov et
al. 2018](#chernozhukov2018GenericMachineLearning)). GATES first sorts
and groups subjects based on their predicted conditional treatment
effect (as a convention, group #1 should contain subjects with lower
predicted conditional treatment effect and group #5 should contain
subjects with higher predicted conditional treatment effect.) Then use
a weighted OLS 

$$Y=\alpha'X_1\sum_k^K\gamma_k*(D-p(X))*1(G_k)+v$$

to test whether the average treatment effect differ between groups (
$\gamma_k$ significantly different for some k). 

The disadvantages of GATES are that (i) it can not assign individual
into a specific group. As it is recommended to execute the procedure
multiple times to eliminate sampling error, a given household might be
assigned to different groups in each iteration. This therefore makes
it difficult to assign a household into a specific group. (ii) it can
not detect homogeneity. Jacob et al. suggested that GATES has
difficulty detecting homogeneous treatment effect ([Jacob et
al.](#jacobCausalInferenceUsing)). They showed that GATES still
indicates heterogeneity, when feeding the simulated data with
homogeneous treatment effect. 

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
women in Hyderabad to create their own small businesses. Hence
business index was chosen to be our first target to analyze.  

Not all the households being surveyed own businesses. Hence when
analyzing business index, the households without any business (in the
past or at the time) were excluded.  

As business index is the weighted average of business related
variables (business assets, profit etc), it is highly correlated to
those variables. For this reason, all those business related variables
were excluded from the dataset. The other index variables
(`women_emp_index_1`, `social_index_1`) were also removed in order to
prevent confusing results as they are just the weighted average of a
set of variables already included in the dataset.  

```{r biz_dataset}
target_index <- "biz_index_all_1"

endline1_biz <- endline1 %>%
  filter(total_biz_1 != 0) %>%
  select(everything(),
         -hhid,
         -contains("biz"),   # exclude business related variables
         -contains("index"), # exclude all index variables
         target_index)       # add back the target index variable
```

1. Heterogeneity of Treament Effect on Business Index

    We use GATES with 5 groups ($k$ equals to 5), the same number as
    Chernozhukov et al. (2018) used in their paper. 

    ![GATES Business
    Index](/blog/img/seminar/apa19-microfinance/biz_gates-1.png) 

    *Figure5-2-1: Sorted Groups Average Treatment Effects for Business
    Index*
    
    The result shows that while the average treatment effect for all
    households is close to zero, there are signs of heterogeneity, at
    least for the fifth and the first group. This supports our
    hypothesis that the treatment effect is different for different
    groups of households. In addition, the result concludes that about
    half of the households have negative treatment effect, which might
    lead to the conditional treatment effect being close to zero on
    average.  

    In spite of the treatment being ineffective on average, the
    microfinance program could still make great impacts if the policy
    targets only the households that have positive treatment effect.

2. The Groups Separating Variables

    Following graph presents 

    ![The Group Separating
    Variables](/blog/img/seminar/apa19-microfinance/biz_varimp-1.png)
    
    *Figure5-1-2: left: Variable Importance predicted by Causal Random
    Forest; right: Vriable Importance predicted by Two-model Approach*
    
    One interesting thing to notice is that expenses related variables
    dominates the results as most of them are highly important to
    separate households with high treatment effect. We could directly
    concludes from the result that household expenditures are
    important factors that determine the effectiveness of our
    treatment. However, the problem is that the relationship of
    expenditures and the success of business makes it difficult to
    exclude the possibility of endogeneity issue. Does the amount of
    expenditures affect how well the business improve from the
    treatment or are the changes in expenditures the results of
    changes in business status? In this case, the two effects needed
    to be separated in order to get reasonable inferences. 

    There are ways to deal with issues of endogeneity. One well known
    method is to construct instrumental variables to mitigate the
    problem. However, as a well defined instrumental variable could 
    not be found in the dataset, this method could not be applied in
    this case. The viable albeit imperfect option left is to just
    delete those expenditures related variables. Hence for the
    following analysis, the dataset with expenditure variables
    (`nondurable_exp_mo_pc_1`, `health_exp_mo_pc_1` etc) excluded will
    be used. 

### Business Index Without Expenditures Variables

```{r biz_dataset_without_exp}
target_index <- "biz_index_all_1"

endline1_biz_noexp <- endline1 %>%
  filter(total_biz_1 != 0) %>%
  select(everything(),
         -hhid,
         -contains("biz"),   # exclude business related variables
         -contains("exp"),   # exclude expenditures related variables
         -contains("index"), # exclude all index variables
         target_index)       # add back the target index variable
```

1. Heterogeneity of Treament Effect on Business Index

    ![GATES Business Index Without Expenditures
    Variables](/blog/img/seminar/apa19-microfinance/biz_noexp_gates-1.png)
    
    *Figure5-2-1: Sorted Groups Average Treatment Effects for Business
    Index without expenditures variables*
    
    The result is similar to the one with expenditure covariates.

2. The Groups Separating Variables

    ![The Group Separating
    Variables](/blog/img/seminar/apa19-microfinance/biz_noexp_varimp-1.png)
    
    *Figure5-2-2: left: Variable Importance predicted by Causal Random
    Forest; right: Vriable Importance predicted by Two-model Approach*

3. Amount of Informal Loans

    ![Scatter Plot
    informal_amt_1](/blog/img/seminar/apa19-microfinance/biz_noexp_trend_informal_amt-1.png)  
    
    *Figure5-2-3: Scatter Plot of Amount of Informal Loans to
    Predicted Conditional Treatment Effect* 
    
    ![Partial Dependence Plot
    informal_amt_1](/blog/img/seminar/apa19-microfinance/biz_noexp_pdp_informal_amt-1.png)  
    
    *Figure5-2-4: Partial Dependence Plot of Amount of Informal Loans
    to Conditional Treatment Effect*
    
4. Adjusted Household Size
    
    ![Scatter Plot
    hhsize_adj_1](/blog/img/seminar/apa19-microfinance/biz_noexp_trend_hhsize_adj-1.png)  
    
    *Figure5-2-5: Scatter Plot of Adjusted Household Size to
    Conditional Treatment Effect*
    
    ![Partial Dependence Plot
    hhsize_adj_1](/blog/img/seminar/apa19-microfinance/biz_noexp_pdp_hhsize_adj-1.png)  
    
    *Figure5-2-6: Partial Dependence Plot of Adjusted Household Size
    to Conditional Treatment Effect*
    
5. Head Age
    
    ![Scatter Plot
    head_age](/blog/img/seminar/apa19-microfinance/biz_noexp_trend_head_age-1.png)  
    
    *Figure5-2-7: Scatter Plot of Head Age to Conditional Treatment
    Effect*

    ![Partial Dependence Plot
    head_age](/blog/img/seminar/apa19-microfinance/biz_noexp_pdp_head_age-1.png)  
    
    *Figure5-2-8: Partial Dependence Plot of Head Age to Conditional
    Treatment Effect*

### Women Empowerment Index

1. Heterogeneity of Treament Effect on Business Index

    ![GATES Women Empowerment
    Index](/blog/img/seminar/apa19-microfinance/woemp_gates_.png)
    *Figure5-3-1: Sorted Groups Average Treatment Effects for Women
    Empowerment Index*

2. The Groups Separating Variables

    ![The Group Separating
    Variables](/blog/img/seminar/apa19-microfinance/woemp_varimp-1.png)
    
    *Figure5-3-2: left: Variable Importance predicted by Causal Random
    Forest; right: Vriable Importance predicted by Two-model Approach*

3. Durable Goods Expenditures

    ![Scatter Plot
    durables_exp_mo_pc_1](/blog/img/seminar/apa19-microfinance/woemp_trend_durables_exp-1.png)
    
    *Figure5-3-3: Scatter Plot of Durable Goods Expenditures on
    Conditional Treatment Effect*
    
    ![Partial Dependence Plot
    durables_exp_mo_pc_1](/blog/img/seminar/apa19-microfinance/woemp_pdp_durables_exp-1.png)  
    
    *Figure5-3-4: Partial Dependence Plot of Durable Goods
    Expenditures to Conditional Treatment Effect*
    
4. Adjusted Household Size

    ![Scatter Plot
    hhsize_adj](/blog/img/seminar/apa19-microfinance/woemp_trend_hhsize_adj-1.png)
    
    *Figure5-3-5: Scatter Plot of Adjusted Household Size on
    Conditional Treatment Effect*
    
    ![Partial Dependence Plot
    hhsize_adj](/blog/img/seminar/apa19-microfinance/woemp_pdp_hhsize_adj-1.png)  
    
    *Figure5-3-6: Partial Dependence Plot of Adjusted Household Size
    to Conditional Treatment Effect*

5. Head Age

    ![Scatter Plot
    head_age](/blog/img/seminar/apa19-microfinance/woemp_trend_head_age-1.png)
    
    *Figure5-3-7: Scatter Plot of Head Age on Conditional Treatment
    Effect* 
    
    ![Partial Dependence Plot
    head_age](/blog/img/seminar/apa19-microfinance/woemp_pdp_head_age-1.png)  
    
    *Figure5-3-8: Partial Dependence Plot of Head Age to Conditional
    Treatment Effect*


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
Banerjee, Abhijit, Esther Duflo, Rachel Glennerster, and Cynthia
Kinnan. 2015. <i>The Miracle of Microfinance? Evidence from a
Randomized Evaluation.</i> American Economic Journal: Applied
Economics, 7 (1): 22-53.  
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
Hitsch, Guenter J. and Misra, Sanjog, <i>Heterogeneous Treatment
Effects and Optimal Targeting Policy Evaluation</i> (January 28,
2018).  
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
