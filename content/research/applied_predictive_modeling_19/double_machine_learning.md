
+++
title = "Implementation of the Double/ Debiased Machine Learning Approach in Python"
date = '2019-06-18'
tags = [ "DML", "treatment effect", "Class19",]
categories = ["Course projects"]
banner = "img/seminar/sample/hu-logo.jpg"
author = "Class of Summer Term 2019"
disqusShortname = "https-wisample-github-io-blog"
description = "Double/ Debiased Machine Learning"
+++


# <center> Double Machine Learning Implementation </center>
<br />
<p><center><small>Christopher Ketzler*, Guillermo Morishige*</small></center></p>

<center>
<p align=justify>
<i>
Abstract:	The aim of this paper is to replicate and apply the approach provided by Chernozhukov et al. (2016) to get the causal estimand of interest: average treatment effect (ATE) $\ \eta_0 $ using Neyman orthogonality and cross-fitting. For observational data, we will estimate the causal relationship between the eligibility and participation in the 401(k) and its effect on net financial assets; as well to apply it to other datasets, to find the effect of the Pennsylvania Reemployment Bonus on the unemployment duration and the effect of smoking on medical costs. As proposed by Chernozhukov’s Double/Debiased Machine Learning (DML) framework, we will estimate the causal effects of binary treatments on an outcome, the regression parameter in a partially linear regression model. With use of machine learning (ML) methods to estimate the nuisance parameters $\ \eta_0 $ : the dependency of the confounding factors (controls) with respect to the outcome and the treatment assignment.

</i>
<br>
<br>

<b>Keywords:</b>	Double machine learning, average treatment effect, Neyman-orthogonality, cross-fitting, partially linear regression model. 
<br>
<br>
* School of Business and Economics, Humboldt Universität zu Berlin, Spandauer Str. 1, 10178, Berlin, Germany.
</p>

</center>

## 1) Introduction

People in the fields of econometrics, epidemiology, philosophy, just to name a few, have been interested into modelling causality: drawing conclusion through statistical analyses from associations between measurements. Although getting inferences from these statistical analyses could be tricky since the association (correlation) doesn’t imply causation. The word “causation” started to appear in settings of randomized experiments by Neyman (1923). Fisher (1935) stressed the importance of randomization as the basis for inference. Rubin (1974) takes it to a non-random assignment mechanism which could apply not only to experimental data, but also observational. 
<br>
As computational power increased, innovations in statistical inference followed. New statistical approaches were developed and now these robust models could handle big sets of data with an extensive number of semi-parametric covariates:
<br>
<br>
In 2016, Victor Chernozhukov et al. introduced the “Double/ Debiased Machine Learning for Treatment and Structural Parameters” to solve the classic semiparametric problem of inference on a low parameter $\ \theta_0 $ in the presence of a high-dimensional nuisance parameter $\ \eta_0 $. A nuisance parameter represents an intermediate step for computing the parameter of interest. In this case, the treatment effect on a certain variable denoted by $\ \theta_0 $ is of interest. Victor Chernozhukov et al. estimate the nuisance parameters through machine learning estimators.
<br>
<br>
Only machine learning models are applicable, which able to handle high dimensional cases, meaning that the entropy of the parameter space is increasing based on the sample size in a sufficiently small way (traditional framework). The following predictors are employed: Random Forest, Lasso, Ridge, Deep Neural Networks, Boosted Trees, and ensemble models based on at least one of them. This approach reduces the effect of easily overfitting and find a suitable trade-off between regularization and bias. By cross-fitting and using Neyman-orthogonal moment functions/ score functions Double/ Debiased Machine Learning reduces the bias, to get a closer estimate for the treatment effect $\ \theta_0 $. Neyman-orthogonal functions have a lower sensitivity with respect to nuisance parameters while estimating treatment effects $\ \theta_0 $.
<br>
<br>
This blog provides a rough overview of Victor Chernozhukov et al. Double Machine Learning approach. We do not aim to clarify all aspects. For further explanations, refer to the paper Chernozhukov et al (2016). The objective of our work is the implementation of Double Machine Learning approach in Python. Therefore, the blog is structured as followed: In section 2) we will make reference to developments in the machine learning field for average treatment estimation purposes. Section 3) provides a deeper insight into DML. In section 4) contains the empirical test of our code and interpretation of results. 
<br>
<br>

## 2) Literature Review
For unconfounded assignment of the treatment effects there are a number of approaches that have been used through the development of statistical inference. Using the inverse of nonparametric estimates of the propensity scores for treatment effect estimations was an idea introduced by Hirano, Keisuke, et al. (2003). Elizabeth Stuart (2010) considered a wide range of matching methods to best compare the treatment effect between groups with covariates in common for an unbiased comparison. Knaus, et. al. (2018) used machine learning to simulate data generation processes (DGPs).
<br>
<br>
	The machine learning estimators have been used for the estimation of heterogenous causal effects across different disciplines. The approaches with the different machine leraning methods are: regression trees by Su, et. al. (2009), random forests by Wagner and Athey (2018), lasso by  Qian and Murphy (2011), support vector machines by Imai and Ratkovic (2013),  boosting by Powers, et. al. (2018), neural networks by Johansson, F., et. al. (2016).
<br>
<br>
	Specifically focused in developments of the Double Machine Learning, we can find an applied study by Knaus (2018): A Double Machine Learning Approach to Estimate the Effects of Musical Practice on Student’s Skills. He used the dataset of the German National Economic Panel Study (NEPS) Blossfeld and von Maurice (2011).
<br>
<br>
	Chernozhukov et al (2016) also provided extensions to the model that are not going t be implemented by us. They proposed using instrumental variables (IV) in the partially linear model. He also estimates the average treatment effect on the treated (ATTE) and the local average treatment effects (LATE).


## 3) Double/ Debiased Machine Learning

### 3.1) Partially Linear Model

The mathematical model that describes the estimation problem is a partially linear equation as suggested by Robinson (1988). It is assumed that the treatment effects are fully heterogenous and the treatment variable is binary, $\ D \in{0,1}$. We consider the vectors $\ (Y,D,X)$, where $\ Y $ are the outcome variables, $\ D $ the treatment variable, and  $\ X $ are the covariates. 
<br>
<br>
<figure role="group">
  $\ Y = D\theta_0 + g_(X) + U,   E[U|X,D]=0 $  
  <figcaption>
     [Eq. 1.1]
  </figcaption>
</figure>
<br>
<figure role="group">
  $\ D = m_0(X)+V,                E[V|X]=0 $   
  <figcaption>
     [Eq. 1.2]
  </figcaption>
</figure>

<br>
<br>

$\ U $ and $\ V $ are disturbances. Our variable of interest is $\ \theta_0 $, the average treatment effect. The nuisance parameter is:  $\ \eta_0 = (m_0,g_0) $. The nuisance parameters are estimated using machine learning methods caused by the nonparametric nature of the variables in the covariates.
<br>
<br>

### 3.2)	Naïve Estimator

A simple way to estimate the treatment effect is to construct a sophisticated machine learning estimator, i.e. Random Forest, and to learn the following regression function: $\ D\theta_0+g_0(X) $. Where the data is split into two parts with $\ i\in I $ and an auxiliary part with length $\ N-n $ . Then one solves the following equation to get the treatment effect:
<br>
<br>
   <figure role="group"> 
  $\ \hat{\theta_0} = (\frac{1}{n} \sum_{i \in I}D_i^2)^{-1} \frac{1}{n} \sum_{i \in I} D_i (Y_i-\hat{g}_0(X_i)). $ 
<figcaption>
     [Eq. 2.1]
  </figcaption>
</figure>
<br>
<br>

<script src="https://gist.github.com/ketzler/c02e8b5904fe7a9f8dd26c0b3e7c4a55.js"></script>

<br>
<br>
By decomposing the scaled estimation error in the treatment effect ($\ \theta_0 $) one can visualize the impact of the bias while learning the ml estimator $\ g_0 $. 
<br>
<br>

<figure role="group">
$\ \sqrt{n}(\hat{\theta}_0 - \theta_o) = \underline{ (\frac{1}{n} \sum_{i \in I} D_i^2)^{-1} \frac{1}{\sqrt{n}} \sum_{i \in I} D_i U_i } + (\frac{1}{n} \sum_{i \in I} D_i^2)^{-1} \frac {1}{\sqrt{n}} \sum_{i \in i} D_i (g_0(X) - \hat{g}_0(X_i))$
<figcaption>
     [Eq. 2.2]
  </figcaption>
</figure>

<br>
<br>
Where term $\ a $ (underlined), under mild conditions, follows: $\ a \to N(\theta, \sum^-) $ and $\ b $ is the regularization bias term. 
<br>
<br>
<figure>
<img align="center"  src="/blog/img/seminar/double_machine_learning/plotEffectDML.PNG">
 <figcaption>Fig.1: The left panel visualizes the simulated distribution of the treatment effect computed by a conventional (non-orthogonal or naive) ML estimator. The estimator is badly biased, meaning that the distribution is shifted to much to the right. The right panel shows the bahavior of an orthogonal, DML estimator, which is unbiased.</figcaption>
<figure>
<br>
<br>
Victor Chernozhukov at al. (2016) provide two algorithms, called DML 1 and DML 2, for computing the nuisance parameter $\ \eta_0 $ based on the prediction of a machine learning model. The authors assume that the true value of the nuisance parameter can be estimated by training a machine learning model on only a part of the data. The algorithms using orthogonalization to compute an orthogonalized regressor for the computation of $ \theta_0 $. 

### 3.3) DML1 - Algorithm

<ol>
    <li>Split the dataset in $\ K $ random folds of observation. For each fold the remaining ones form the complement set. The recent fold is called main set/fold. 
    <li>For each fold create a machine learning estimator and train it with the complement set. The kind of the model is derived by the conditions below.
    <li>Construct for each main fold and the depending estimator the theta estimator. Each fold provides an own theta value. One could use one of the following two theta estimators below. The estimators solve the following equattion:  
    <br>
    <br>
    <figure>
    $\ \sum_{k=1}^K E_{n,k} [ \varphi (W; \tilde{ \theta }_0 , \hat{\eta}_{0,k}) ] \ = 0 $
        <figcaption>
         [Eq. 3.1]
        </figcaption>
      </figure>
    <li>Compute the mean over the theta values.
        <br>
        <br>
        <figure>
        $\ \tilde{ \theta}_0 = \frac{1}{K} \sum_{k=1}^K \hat{\theta}_{o,k} $
        <figcaption>
         [Eq. 3.2]
        </figcaption>
      </figure>
        <br>
        <br>
        
 
 The theta estimator consists of a term for partialling out the effect of $\ X $ from $\ D $ to compute the orthogonalized regressor $\ V=D-m_0(X) $. Nuisance parameter $\ m_0(X) $ represents a machine learning model, which is trained with the complement set of the DML algorithms. This auxiliary prediction problem to estimate the conditional mean of $\ D $ given $\ X $ is called “double prediction” or “double machine learning” by the authors.
 <br>
 <br>
 Two theta estimators are provided by the authors. Both are computed on the main sample of the data. 
 <br>
        <br>
        <figure>
        $\ \hat{\theta}_0 = (\frac{1}{n} \sum_{i \in I} \hat{V}_i D_i)^{-1} \frac{1}{n} \sum_{i \in I} \hat{V}_i( Y_i - \hat{g}_0(X_i)) $
        <figcaption>
         [Eq. 4.1]
        </figcaption>
        </figure>
        <br>
        <br>
        or
        <br>
        <br>
        <figure>
        $\ \hat{\theta}_0 = (\frac{1}{n} \sum_{i \in I} \hat{V}_i \hat{V}_i)^{-1} \frac{1}{n} \sum_{i \in I} \hat{V}_i( Y_i - \hat{l}_0(X_i)) $, with $\ l_0 = E[Y|X] $
        <figcaption>
         [Eq. 4.2]
        </figcaption>
        </figure>
        <br>
        <br>
 By the orthogonalization of $\ D $ with respect to $\ X $, the direct effect of confounding is partially removed by subtracting the nuisance estimation of $\ g_0 $ . The $\ \theta_0 $ reduces the effect of the regularization bias that is affecting (3.3). This theta estimator can clearly be interpreted as a linear instrumental variable estimator. The second approach is based on Robinson (1998) for the representation of a debiased estimator.
 <br>
 <br>
<script src="https://gist.github.com/ketzler/45caf1a96c4bf3e243b9c916e5add0b4.js"></script>
 <br>
 <br>
 The second DML algorithm works similar, but it doesn’t have to aggregate single theta values. This one recommended over DML1, because the pooled empirical Jacobian behave more stable. The optimal number of folds is four or five. The lead to better results than for example two. 
 <br>
 <br>
 
 ### 3.4) DML2 - Algorithm
 
 <ol>
    <li>Split the dataset in $\ K $ random folds of observation. For each fold the remaining ones form the complement set. The recent fold is called main set/ fold. 
    <li>For each fold create a machine learning estimator and train it with the complement set. The kind of the model is derived by the conditions below.
    <li>Construct one theta estimator as a solution of the following equation.
        <br>
        <br>
        <figure>
        $\ \frac{1}{K} \sum_{k=1}^K E_{n,k} [ \varphi (W; \tilde{ \theta }_0 , \hat{\eta}_{0,k}) ] \ = 0 $
        <figcaption>
         [Eq. 5]
        </figcaption>
        </figure>

<br>
<br>
     To find the best estimator of all of them we will consider the ones that give the least squared errors. The combination of machine learning methods for each part of the two theta estimations could yield better results, since some of these thetas could be better estimated with one kind of machine learning method.
     <br>
     <br>
     
### 3.5) Sample Splitting

Sample splitting guarantees an efficient remove of the bias induced by overfitting. After the split Y and D are estimated within the main sample. The effect of D in V ($\ V = D - m(X) $) is left out so that the $ \hat{ \theta} $ term doesn't cause a bias. Then on the remaining samples (folds), the $ \hat{ \theta} $ is calculated following formula (4.1, 4.2). By aggregation of the single theta values over the main and auxiliary samples, the algorithm reaches full efficiency in predicting the treatment effect. To prevent efficiency loss, the procedure is repeated alternating the folds estimating the D and Y and the folds estimating for $ \hat{ \theta} $. Victor Chernozhukov et al. (2016) called this procedure cross-fitting. 
<br>
<br>
<figure>
<img align="center" src="/blog/img/seminar/double_machine_learning/SampleSplitting.PNG">
 <figcaption>Fig.2: The left panel shows an overfitted estimator, which theta distribution is shifted. The other panel visualizes a distribution without any kind of bias based on sample splitting.</figcaption>
<figure>
<br>
<br>
The following figure describes the effect of cross-fitting on the performance. The left plot visualizes the finite-sample distribution of the treatment effect computed with the partially linear model of Robinson 1998 without sample splitting. The right side illustrates the same case with sample splitting. One can easily realize that the bias is eliminated. 
<br>
<br>
For more detailed approaches like moment condition models with linear scores or non-linear scores, please refer to the original paper of Victor Chernozhukov et al. (2016).
<br>
<br>


## 4) Empirical Examples
<br>
In the following section we will provide an overview over three datasets we use to evaluate our implementation. Two of them are used in the paper of Chernozhukov et al (2016), so we are able to compare our results with the ones done in the original paper. The third one is a Kaggle dataset. Because of the long computation time we abstain of parameter tuning and fit the models based on their standard properties. We compared the naïve approach and the DML 1 algorithm based on neural networks, decision trees, and Lasso. We split the data set into four folds. 
<br>
<br>

### The 401 (k) plan

<br>
The purpose for the average treatment effect is to find the impact of participating in the 401 (k) plan with respect to the net financial assets Y. 401(k) plan denotes a pension account in the United States. Participants pay their contributions based on their income. Not each firm is offering this kind of retirement and the participation is free. The idea beyond this dataset is, that when the 401(k) plan program started, people would make job decisions not based on the retirement offers, hence, on the income or other job criteria. The main problem of the experimental dataset is that it lacks random assignment. To partial that out and consider the data as exogenous there must be done some conditioning on income and other variables related to job choice that might have an association with whether the firm has the 401 (K) plan available, based on  the argument of Poterba et. al. (1994). The dataset based on the Survey of Income and Program Participation of 1991. In our covariates we can find: Age of the head of the household, household income, household size, years of education of the head of the household, married indicator, two-earner status indicator, house ownership indicator, etc. 
<br>
<br>
If we compare our results with the results of the original paper. Our treatment effects are too high. Also, the results of the original paper are close next to each other, our results differ too much. This could be caused by the missing parameter tuning. Our implementation also leads to a relativ high standard deviation, meaning that there is a high bias in our models, which should be eliminated by the DML algorithm.
<br>
<br> 

 <table style="width:100%">
  <tr>
    <th>Given Estimator</th>
    <th>Nuisance Estimator</th>
    <th>Mean</th>
    <th>Median</th>
    <th>Standard deviation</th>
  </tr>
  <tr>
    <td>Naïve Estimator</td>
    <td>Random Forest</td>
    <td>13226.91</td>
    <td>13196.08</td>
    <td>1878.16</td>
  </tr>
  <tr>
    <td>Neural Network</td>
    <td>Neural Network</td>
    <td>-5577.51</td>
    <td>1111.23</td>
    <td>128477.81</td>
  </tr>
   <tr>
    <td>Dictionary Learning</td>
    <td>Lasso</td>
    <td>5506.19</td>
    <td>5532.03</td>
    <td>114.51</td>
  </tr>    
   <tr>
    <td>Decision Tree</td>
    <td>Decision Tree</td>
    <td>16872.88</td>
    <td>16830.74</td>
    <td>1752.67</td>
  </tr>
   <tr>
    <td>Extra Trees</td>
    <td>Decision Tree</td>
    <td>16845.95</td>
    <td>16927.08</td>
    <td>1784.50</td>
  </tr>
</table> 
<br>
<br>
All treatment effect distributions plots show a gaussian distribution. Our neural network seems to overfit with some large outliers, which effect the median and mean in a strong way. The Decision tree used in the Extra Trees case generates a well distributed gaussian distribution. If one compares this plot with the plot of the naïve Estimator, one realizes that the distributions are shifted by 4000. We think that a better parameter tuning could solve this problem.

<table style="width:100%">
  <tr>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/Naive401kplan.png">
 <figcaption>Fig.3.1: treatment effect distribution: Naive Estimator</figcaption>
      </td>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/DictionaryLearning, DML1 - 401(k) plan.png">
 <figcaption>Fig.3.2: treatment effect distribution: given Estimator: Dictionary Learning; Nuisance Estimator: Lasso </figcaption>
      </td>
     <\tr>
         <tr>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/DecisionTreeRegressor, DML1 - 401(k) plan.png">
 <figcaption>Fig.3.3: treatment effect distribution: given Estimator: Decision Tree; Nuisance Estimator: Decision Tree</figcaption>
      </td>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/MLPRegressor DML1 401(k) plan.png">
 <figcaption>Fig.3.4: treatment effect distribution: given Estimator: Neural Network; Nuisance Estimator: Neural Network</figcaption>
      </td>
     <\tr>
         <tr>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/ExtraTreesRegressor, DML1 - 401(k) plan.png">
 <figcaption>Fig.3.5: treatment effect distribution: given Estimator: ExtraTreesRegressor (Ensemble); Nuisance Estimator: Decision Tree </figcaption>
      </td>
</tr>
</table>

### Pennsylvania Reemployment Bonus experiment dataset

<br>
We try to find the effect of the unemployment bonus on unemployment duration. This dataset was used in the original paper of Chernozhukov et al (2016). The Pennsylvania Reemployment Bonus experiment dataset contains observation of testing the incentive effects of alternative compensation schemes for unemployment insurance by the US Department of Labor in the 1980’s which was previously analyzed by Bilias (2000) and Bilias and Koenker (2002). Five groups of unemployed persons were defined. One control group, which still get treated by the standard rules of unemployment insurance, and four treated groups, all randomly assigned. Treated persons got a cash bonus, if they got hired. The cash bonus differed by some criteria. The given dataset only contains persons, who get treated by the treatment group 4. This group received a high bonus amount, but they were forced to take part in a longer qualification period. We only consider the 4th group and the control, to have a binary treatment variable D. Participants could take part in special Workshops.  The outcome variable tries to catch the effect of this support on the duration of the unemployment period, and its vector of covariates includes age, gender, race, number of dependents, location, type of occupation, etc. 
<br> 
<br>
Our approach performed better on this data set. Our results are close to the of the original paper. Only small differences between the median and mean refer a low number of outliers and better predictions. In this scenario the decision trees and neural network seem to perform best by providing results close to the original papers. Parameter tuning should increase performance and accuracy. This data set is larger than the 401(k) plan data set, that could cause the better performance.
<br>
<br>
 <table style="width:100%">
  <tr>
    <th>Given Estimator</th>
    <th>Nuisance Estimator</th>
    <th>Mean</th>
    <th>Median</th>
    <th>Standard deviation</th>
  </tr>
  <tr>
    <td>Naïve Estimator</td>
    <td>Random Forest</td>
    <td>-0.090</td>
    <td>-0.092</td>
    <td>0.046</td>
  </tr>
  <tr>
    <td>Neural Network</td>
    <td>Neural Network</td>
    <td>-0.085</td>
    <td>-0.084</td>
    <td>0.010</td>
  </tr>
   <tr>
    <td>Dictionary Learning</td>
    <td>Lasso</td>
    <td>-0.104</td>
    <td>-0.104</td>
    <td>0.001</td>
  </tr>    
   <tr>
    <td>Decision Tree</td>
    <td>Decision Tree</td>
    <td>-0.092</td>
    <td>-0.091</td>
    <td>0.023</td>
  </tr>
   <tr>
    <td>Extra Trees</td>
    <td>Decision Tree</td>
    <td>-0.084</td>
    <td>-0.087</td>
    <td>0.021</td>
  </tr>
</table> 
<br>
<br>
Only the naïve Estimator does not generate a gaussian distribution. The remaining estimators behave as expected. Only the Lasso model computes values, which are ten times higher then of the rest. We do not have an explanation for this result. The DML 1 algorithm based on a Decision tree or neural network seems to perform best.
<br>

<table style="width:100%">
  <tr>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/Naive - Pennsylvania.png">
 <figcaption>Fig.4.1: treatment effect distribution: Naive Estimator</figcaption>
      </td>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/DictionaryLearning DML1 Pennsylvania.png">
 <figcaption>Fig.4.2: treatment effect distribution: given Estimator: Dictionary Learning; Nuisance Estimator: Lasso </figcaption>
      </td>
     <\tr>
         <tr>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/DecisionTreeRegressor DML1 Pennsylvania.png">
 <figcaption>Fig.4.3: treatment effect distribution: given Estimator: Decision Tree; Nuisance Estimator: Decision Tree</figcaption>
      </td>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/MLPRegressor DML1 Pennsylvania.png">
 <figcaption>Fig.4.4: treatment effect distribution: given Estimator: Neural Network; Nuisance Estimator: Neural Network</figcaption>
      </td>
     <\tr>
         <tr>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/ExtraTreesRegressor DML1 Pennsylvania.png">
 <figcaption>Fig.4.5: treatment effect distribution: given Estimator: ExtraTreesRegressor (Ensemble); Nuisance Estimator: Decision Tree </figcaption>
      </td>
</tr>
</table>

### The Medical Cost Personal Dataset 

<br>
This dataset is used in Brett Lantz’ (2013) book “Machine Learning with R”. The dataset was published on Kaggle and describes the medical costs by different persons based on features like the body mass index, sex, smoker, etc. We consider, whether a person is smoking as treatment variable and compute the effect of smoking on the medical costs incurred.
<br>
<br>
This data set is not part of the original paper. Therefor, we only can guess how good our algorithm performs. All results are close next each other. We interpret this as a good sign for the performance and accuracy. The median and mean values are close together, therefore we expect a low amount of outliers.  
<br>
<br>
 <table style="width:100%">
  <tr>
    <th>Given Estimator</th>
    <th>Nuisance Estimator</th>
    <th>Mean</th>
    <th>Median</th>
    <th>Standard deviation</th>
  </tr>
  <tr>
    <td>Naïve Estimator</td>
    <td>Random Forest</td>
    <td>24392.85</td>
    <td>24366.45</td>
    <td>949.12</td>
  </tr>
  <tr>
    <td>Neural Network</td>
    <td>Neural Network</td>
    <td>31347.34</td>
    <td>31324.24</td>
    <td>1147.35</td>
  </tr>
   <tr>
    <td>Dictionary Learning</td>
    <td>Lasso</td>
    <td>23870.46</td>
    <td>23841.01</td>
    <td>94.51</td>
  </tr>    
   <tr>
    <td>Decision Tree</td>
    <td>Decision Tree</td>
    <td>31483.46</td>
    <td>31344.05</td>
    <td>1082.80</td>
  </tr>
   <tr>
    <td>Extra Trees</td>
    <td>Decision Tree</td>
    <td>31502.27</td>
    <td>31541.10</td>
    <td>1171.23</td>
  </tr>
</table> 
<br>
<br>
The generated plots behave as expected. If one compares the naïve Estimator with the remaining one, one can realize the eliminated bias by the DML algorithm. We think that the estimators fitted very well on this data set. 
<br>
<br>

<table style="width:100%">
  <tr>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/Naive - Insurance.png">
 <figcaption>Fig.5.1: treatment effect distribution: Naive Estimator</figcaption>
      </td>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/DictionaryLearning, DML1 - Insurance.png">
 <figcaption>Fig.5.2: treatment effect distribution: given Estimator: Dictionary Learning; Nuisance Estimator: Lasso </figcaption>
      </td>
     <\tr>
         <tr>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/DecisionTreeRegressor, DML1 - Insurance.png">
 <figcaption>Fig.5.3: treatment effect distribution: given Estimator: Decision Tree; Nuisance Estimator: Decision Tree</figcaption>
      </td>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/MLPRegressor, DML1 - Insurance.png">
 <figcaption>Fig.5.4: treatment effect distribution: given Estimator: Neural Network; Nuisance Estimator: Neural Network</figcaption>
      </td>
     <\tr>
         <tr>
      <td>
<img align="center" width=50% src="/blog/img/seminar/double_machine_learning/ExtraTreesRegressor, DML1 - Insurance.png">
 <figcaption>Fig.5.5: treatment effect distribution: given Estimator: ExtraTreesRegressor (Ensemble); Nuisance Estimator: Decision Tree </figcaption>
      </td>
</tr>
</table>

## 5) Conclusion
<br>
This blog post gave a fast introduction into the Double machine learning approach. We explained the basic idea and refer for further readings to the original paper of Victor Chernozhukov et al. and implemented parts of the approach in Python. By applying the code on the different data sets we evaluated our work. The next steps on this work would be the implementation of the DML2 algorithm and the extends named in the paper. The apllying of the algorithms on other data sets and the extension with other machine learning models. Also we would use parameter tuning for enlarging accuracy. 

<br>
<br>
The entire code:
<br>
<script src="https://gist.github.com/ketzler/e0296157edba4a1ad62d8c7df6918832.js"></script>

### References
<br>
[01]
<i> Abadie, Alberto. <i\> 
<b> “Semiparametric Instrumental Variable Estimation of Treatment Response Models.” </b>  
Journal of Econometrics, vol. 113, no. 2, 2003, pp. 231–263., doi:10.1016/s0304-4076(02)00201-4.
    <br> <br>
    [02]
<i>Bilias, Yannis.<i\> <b>“Sequential Testing of Duration Data: the Case of the Pennsylvania ‘Reemployment Bonus’ Experiment.”</b>  Journal of Applied Econometrics, vol. 15, no. 6, 2000, pp. 575–594., doi:10.1002/jae.579.
    <br> <br>
    [03]
<i>Bilias, Yannis, and Roger Koenker.<i\><b> “Quantile Regression for Duration Data: A Reappraisal of the Pennsylvania Reemployment Bonus Experiments.”</b>  Economic Applications of Quantile Regression, 2002, pp. 199–220., doi:10.1007/978-3-662-11592-3_10.
    <br> <br>
    [04]
<i>Blossfeld, Hans-Peter, and Jutta Von Maurice.<i\><b> “Education as a Lifelong Process.”</b>  Zeitschrift Für Erziehungswissenschaft, vol. 14, no. S2, 2011, pp. 19–34., doi:10.1007/s11618-011-0179-2.
    <br> <br>
    [05]
<i>Chernozhukov, Victor, et al.<i\><b> “Double/Debiased Machine Learning for Treatment and Structural Parameters.”</b>  2016. Econometrics Journal, vol. 21, no. 1, pp. C1–C68., doi:10.1111/ectj.12097. 
<br> <br>
    [06]
<i>Chernozhukov, Victor, et al.<i\><b> “Double/Debiased/Neyman Machine Learning of Treatment Effects .”</b>  2017. American Economic Review, vol. 107, no. 5, pp. 261–265., doi:10.1257/aer.p20171038.
    <br> <br>
    [07]
<i>Fisher, R. A., and Harold Hotelling.<i> <b>“The Design of Experiments.”</b>  Journal of the American Statistical Association, vol. 30, no. 192, 1935, p. 771., doi:10.2307/2277749.
    <br> <br>
    [08]
<i>Glymour, Clark.<i\> <b>“Causation and Statistical Inference.”</b> The Oxford Handbook of Causation, Edited by Helen Beebee et al., Nov. 2010, doi:10.1093/oxfordhb/9780199279739.003.0024.
    <br> <br>
    [09]
<i>Hirano, Keisuke, et al.<i\><b> “Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score.”</b>  Econometrica, vol. 71, no. 4, 2003, pp. 1161–1189., doi:10.1111/1468-0262.00442.
<br> <br>
    [10]
<i>Hitchcock, Christopher.<i\><b> “Causal Modelling.”</b>  The Oxford Handbook of Causation, Edited by Helen Beebee et al., Nov. 2009, doi:10.1093/oxfordhb/9780199279739.003.0015.
    <br> <br>
    [11]
<i>Imai, Kosuke, and Marc Ratkovic.<i\><b> “Estimating Treatment Effect Heterogeneity in Randomized Program Evaluation.”</b>  The Annals of Applied Statistics, vol. 7, no. 1, 2013, pp. 443–470., doi:10.1214/12-aoas593.
<br> <br>
    [12]
<i>Imbens, Guido W., and Donald B. Rubin.<i\><b> "Causal Inference for Statistics, Social, and Biomedical Sciences: an Introduction."</b>  Cambridge University Press, 2015.
<br> <br>
    [13]
<i>James, Gareth, et al.<i\><b> "An Introduction to Statistical Learning: with Applications in R."</b>  Springer, 2017.
    <br> <br>
    [14]
<i>Johansson, F., et. al.<i\><b> “Learning representations for counterfactual inference.”</b>  In International conference on machine learning, 2016, pp. 3020–3029.
    <br> <br>
    [15]
<i>Knaus, Michael C.<i\><b> “A Double Machine Learning Approach to Estimate the Effects of Musical Practice on Student’s Skills.”</b>  2018. Swiss Institute for Empirical Economic Research (SEW), University of St. Gallen.
<br> <br>
    [16]
<i>Knaus, Michael C., et. al.<i\><b> “Machine Learning Estimation of Heterogeneous Causal Effects: Empirical Monte Carlo Evidence”</b>  2018. Swiss Institute for Empirical Economic Research (SEW), University of St. Gallen.
<br> <br>
    [17]
<i>Lantz, Brett.<i\><b> "Machine Learning with R: Learn How to Use R to Apply Powerful Machine Learning Methods and Gain an Insight into Real-World Applications."</b>  Packt Publishing, 2013.
    <br> <br>
    [18]
<i>Neyman, J.<i\> <b>“c(α) tests and their use.”</b>  Sankhyā: The Indian Journal of Statistics, Series A (1961-2002), 1 July 1979, vol. 41, pp.1-21.
    <br> <br>
    [19]
<i>Poterba, Venti, and Wise.<i\> <b>“Do 401(k) contributions crowd out other personal saving?”</b>  Journal of Public Economics 58, 1994, pp. 1–32.
    <br> <br>
    [20]
<i>Poterba, Venti, and Wise.<i\><b> “401(k) plans and tax-deferred savings.”</b>  Studies in the Economics of Aging, Chicago: University of Chicago Press, 1994, pp. 105–142.
    <br> <br>
    [21]
<i>Powers, Scott, et al.<i\><b> “Some Methods for Heterogeneous Treatment Effect Estimation in High Dimensions.”</b>  Statistics in Medicine, vol. 37, no. 11, 2018, pp. 1767–1787., doi:10.1002/sim.7623.
<br> <br>
    [22]
<i>Qian, Min, and Susan A. Murphy.<i\><b> “Performance Guarantees for Individualized Treatment Rules.”</b>  The Annals of Statistics, vol. 39, no. 2, 2011, pp. 1180–1210., doi:10.1214/10-aos864.
    <br> <br>
    [23]
<i>Robinson, P. M.<i\><b> “Root-N-Consistent Semiparametric Regression.”</b>  Econometrica, vol. 56, no. 4, 1988, p. 931., doi:10.2307/1912705.
    <br> <br>
    [24]
<i>Rubin, Donald B.<i\> <b>“Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies.”</b>  Journal of Educational Psychology, vol. 66, no. 5, 1974, pp. 688–701., doi:10.1037/h0037350.
    <br> <br>
    [25]
<i>Speed, T. P.<i\> <b>“Introductory Remarks on Neyman (1923).”</b>  Statistical Science, vol. 5, no. 4, 1990, pp. 463–464., doi:10.1214/ss/1177012030.
    <br> <br>
    [26]
<i>Stuart, Elizabeth A.<i\><b> “Matching Methods for Causal Inference: A Review and a Look Forward.”</b>  Statistical Science, vol. 25, no. 1, 2010, pp. 1–21., doi:10.1214/09-sts313.
    <br> <br>
    [27]
<i>Su, Xiaogang, et al.<i\><b> “Subgroup Analysis via Recursive Partitioning.”</b>  SSRN Electronic Journal, 2009, doi:10.2139/ssrn.1341380.
    <br> <br>
    [28]
<i>Wager, Stefan, and Susan Athey.<i\><b> “Estimation and Inference of Heterogeneous Treatment Effects Using Random Forests.”</b>  Journal of the American Statistical Association, vol. 113, no. 523, 2018, pp. 1228–1242., doi:10.1080/01621459.2017.1319839.
    <br> <br>
    [29]
<i>Fan, Jianqing, et al. <i\><b>“Variance Estimation Using Refitted Cross-Validation in Ultrahigh Dimensional Regression.”</b>  Journal of the Royal Statistical Society: Series B (Statistical Methodology), vol. 74, no. 1, 2011, pp. 37–65., doi:10.1111/j.1467-9868.2011.01005.x.


```python

```
