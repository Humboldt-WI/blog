+++
title = "Group 3t"
date = '2017-12-14'
tags = [ "Deep Learning", "Neural Networks", "Class19/20",]
categories = ["Course projects"]
banner = "img/seminar/sample/hu-logo.jpg"
author = "Class of Winter Term 2017 / 2018"
disqusShortname = "https-wisample-github-io-blog"
description = " group 3"
+++

# Introduction [¶](#introduction)

It's no secret that consumers today are faced with a more product choices than ever before. While having many choices can be beneficial to a point, eventually it can become overwhelming and consumers can become paralysed in the face of so many options. For this reason, many online, and offline, businesses have created tools and methods to recommend certain products to certain groups of consumers, thereby reducing the number of products a consumer has to consider. Of course, recommendation is no easy task, and much has been written about the best method to use.

Many recommendation algorithms focus on latent variables that seek to describe underlying user preferences and item attributes. Most, however, do not include external factors such as price or seasonality. From a retailer's perspective, it would be ideal to know the maximum price each customer is willing to pay for each item. Retailers could then potentially have personalized per-item prices for each consumer, which would maximize the retailer's revenue.

One way this could be achieved is if all products were unpurchasable by default, or simply more expensive than anyone's willingness to pay (WTP), and all customers entering the store receive a voucher for all the products they came to buy, allowing them to purchase at the price that they are willing to pay. Though unrealistic, this example illustrates how one could achieve personalized pricing, as actively displaying different prices per customer in a store is unfeasible, and if possible, would lead to customer dissatisfaction.

In this blog, we will build an interpretable network to predict customer choice. We will estimate latent user preferences, seasonal effects and price sensitivities. The focus will be placed particularly on the grocery industry. Firstly we review the current state-of-the-art models used for personalized recommendation: Poisson factorization and Exponential family embeddings. Then we showcase a model combining and further extending the features of these two models, a sequential probabilistic model called SHOPPER. Finally, we proceed to build our own predictive model based on SHOPPER, through embeddings on customer, item, basket, seasonal and price levels.

## Status Quo in Marketing [¶](#status-quo-in-marketing)

### Poisson factorization [¶](# poisson-factorization)

Widely used in E-commerce settings, hierarchical Poisson factorization captures user-to-item interaction, extracting user preferences for recommendation of previously unpurchased products through past purchases of other similar customers. The mathematics underlying this model are straightforward: users' preferences are Gamma distributed based on their past activity (number of shopping trips) and items' attributes are Gamma distributed based on their popularity (number of purchases). The user-item interaction is expressed in the rating (either a points scale, such as 1-5 stars, or binary dummy for purchase), which is Poisson distributed according to the user's preference and item's attribute. As a simple example, imagine a customer who has only bought vegetables in her five previous shopping trips (activity). This customer's preference would thus be classified as vegetarian. On the other hand, the shop has a variety of frequently bought (popular) vegetables (which is an attribute of those items). As other customers with the preference "vegetarian" have bought asparagus, the resulting "rating" will be high, and thus this customer would receive a recommendation to buy asparagus.
It is called hierarchical due to the ordering of both items and users; some products are bought more frequently than others (are more popular), and some customers execute more purchases than others (are more active).

### Exponential Family Embeddings [¶](# exponential-family-embeddings)

Yet better than Poisson factorization is the collection of models called Exponential family embeddings. It stems from methods created for natural language processing, such as word embeddings, but can be extended to other highly-relational datasets. It is composed of individual terms (originally words, in our case grocery items) that make up a corpus (dictionary or store inventory) and are mutually bound in a context (sentence or a shopping basket) via conditional probability. This link is the probability that a specific item is in this particular basket given all the other items in the same basket, and comes from an exponential distribution, such as Poisson for discrete and Gaussian for real-valued data points. The objective is to maximize these conditional probabilities of the whole corpus, which creates shared embeddings for each term. Through these latent embeddings, we can calculate similarity (and dissimilarity) as the cosine distance between the embedding vectors and thus represent how similar items are to each other. Moreover, "Poisson embeddings can capture important econometric concepts, such as items that tend not to occur together but occur in the same contexts (substitutes) and items that co-occur, but never one without the other (complements)" [p.2, Rudolph et al 2016].

These state-of-the-art models have their shortcomings; neither of these capture price effects and seasonality patterns. This is where our star model, Shopper, jumps in, combining the item interaction of EFE and user preferences of HPF, but also extracts unobserved latent variables (whims), and additionally accounts for price effects and seasonality patterns.

# Shopper - What is it and how does it work? [¶](#shopper---what-is-it-and-how-does-it-work)

The SHOPPER algorithm (_Blei et al._) is a sequential probabilistic model for market baskets. The main idea of SHOPPER is to model how customers choose items in their basket when shopping for groceries, taking various factors into account, including seasonal effects, personal preferences, price sensitivities and cross item interactions. SHOPPER is a structered model, in that it assumes customers behave in a specific way. The strcuture comes form microeconomic theory and a maximazation of utilities(REF).

SHOPPER posits that a customer walks into the store and sequentially chooses each item to be placed in the basket. The customer chooses each item based on personal preferences, price, seasonality and overall item popularity. As each item is placed into the basket, the customer then takes the current products in the basket into account when deciding on the next product. One can imagine a typical shopping trip, where the customer goes in with an intent to buy a specific item, eg cheese, and later decides to spontaneously purchase complementary products, such as wine and crackers, to accompany the cheese. Although the customer initially had no intention of purchasing wine and crackers, the presence of cheese increased the attractiveness of the complementary products. SHOPPER generates latent variables to model these factors, estimates an overall utility value for each shopping trip and calculates a probability of observing a set of items in a given basket.

The basics of the model can be summarized with the following 5 equations:

1. $$\psi_{tc} = \lambda_{c} + \theta_{ut}^T \alpha_{c} - \gamma_{ut}^T \beta_{c} log(r_{tc}) + \delta_{wt}^T \mu_{c}$$
2. $$\Psi(c, y_{t, i-1}) = \psi_{tc} + \rho_{c}^T(\frac{1}{i-1} \displaystyle\sum_{j=1}^{i-1} \alpha_{y_{tj}} )$$
3. $$max \space U_{t,c}(y_{t, i-1}) = \Psi(c, y_{t, i-1}) + \epsilon_{t,c}$$
4. $$p(y_{ti} = c | \mathbf{y}_{t, i-1}) = \frac{exp\{\Psi(c, \mathbf{y}_{t, i-1})\}}{\Sigma_{c' \notin y_{t, i-1}}exp\{\Psi(c', \mathbf{y}_{t, i-1})\}}$$
5. $$\widetilde{U}_t(\mathcal{Y_t}) = \displaystyle\sum_{c \in \mathcal{Y_t}} \psi_{tc} + \frac{1}{|\mathcal{Y_t}| - 1} \displaystyle\sum_{(c, c') \in \mathcal{Y_t} \times \mathcal{Y_t}:c'\neq c} v_{c',c}$$

Although daunting at first, the model can be broken down into smaller pieces to fully understand it. From a conceptual point of view SHOPPER posits that a consumer seeks to maximize the utility received from their basket each shopping trip (equation 5). They do this by selecting an item from all of the available items in the store (equation 4), and the product most likely to be chosen is the one that provides them with the highest utility (equation 3). The utility function can be described as a log-linear function with latent variables representing: item popularity, user preferences, price sensitivities, seasonal effects and cross item interactions (equations 1 and 2). The following sections will delve further into detail for each equation.

## Equation 1 (Latent Variables) [¶](#equation-1-latent-variables)

$$\psi_{tc} = \lambda_{c} + \theta_{ut}^T \alpha_{c} - \gamma_{ut}^T \beta_{c} log(r_{tc}) + \delta_{wt}^T \mu_{c}$$

Equation 1 represents the utility a customer gets from item $c$ in trip $t$. The above equation can be divided into several smaller portions.

_Item Popularity_ : $$\lambda_{c}$$ can be thought of as representing a latent (time invariant) intercept term that captures overall item popularity. In this case, the more popular an item is, the higher the value of this variable should be.

_User Preferences_ : To get a more accurate utility estimate, SHOPPER creates a per-user latent vector $$\theta_{u}$$, along with a per-item latent $$\alpha_c$$. The inner product of the two vectors is then taken and added to the unnormalized log probability of each item. Unlike traditional multinomial logit models, that restrict themselves to a handful of items, or group items to a handful of categorys, using SHOPPER we can generate individual preferences per item for each user individually. A larger inner product indicates a higher preference a user has for a certain item. It is important to note that the per-user latent vector is estimated for each trip, but the per-item vector is held constant across all trips. This represents the fact that  users can have different utilities for an item depending on that individual trip, but the underlying characteristics of the product remain the same across time. An additional benefit to this is that we now also have per-user and per-item vectors that can be used in other tasks like clustering.

_Price Sensitivities_ : An important aspect in discrete choice modelling is the effect of price on choices. Let $r_{tc}$ represent the price of price of product $c$ at trip $t$. SHOPPER posits that each user has an individualized price elasticity to each item. $\gamma_{ut}$ represents a new set of per-user latent vectors that are different from $\theta_{u}$ and $\beta_{c}$ represents a new set of per-item latent vectors that are different from $\alpha_c$. $\gamma_{ut}^T \beta_{c}$ then represents the price elasticity for each user for each item. $\gamma_{ut}^T \beta_{c}$ is restricted to be positive, so that $$- \gamma_{ut}^T \beta_{c} log(r_{tc})$$ remains negative. This implies that an increase in the price of item $c$ will decrease the utility the customer recieves from item $c$, making it less likely the customer chooses item $c$ in trip $t$. This assumption is consistent with similar assumptions in other discrete choice models.

Instead of using the raw price SHOPPER uses the normalized prices, i.e., the price of item $c$ at trip $t$ is divided by the per-item mean price. This has the benefit of bringing $\beta_c$ to a comparable scale, so that items with comparatively larger prices can be compared fairly with items that have a comparatively smaller price. Another benefit is that normalizing the price ensures other latent variables return the average outcome. When the price term takes it's average value, the price term vanishes because $log(1) = 0$

_Seasonality_ : When estimating other latent variables, the effect of seasonality must also be considered. It can be expected that certain items sell more depending on the time of year, e.g. chocolates at Easter, champagne at New Year's, candy at Halloween, etc, partly because demand increases, as well as sales implemented by the store. Neglecting seasonality can lead to estimating variables that do not isolate the effect of price. For example, champagne sells more at New Year's partly because of an increase in demand, but also because the store may put that product on sale to induce more sales. In order to prevent this seasonal effect from confounding the estimates of the other latent vectors, SHOPPER adds the per-week item interaction effect $\delta_{wt}^T \mu_{c}$, where $\delta_{w}$ represents the per-week latent vectors and $\mu_{c}$ represents a new set of per-item latent vactors. The inner product represents the seasonal effect on each product, i.e. products with a higher value in a given week add more to utility than products with a lower value and have a higher probabliity of being purchased. Including $\mu_{c}$ also allows for correlated seasonal effects accross items, i.e. two products that are purchased in the same season should have similar latent vectors. In this model, we could expect chocolate and eggs to have a higher inner product during easter than any other time in the year.

## Equation 2 (Item-Item Interactions) [¶](#equation-2-item-item-interactions)

$$\Psi(c, y_{t, i-1}) = \psi_{tc} + \rho_{c}^T(\frac{1}{i-1} \displaystyle\sum_{j=1}^{i-1} \alpha_{y_{tj}} )$$

Using equation 1 we can calculate the utility a customer receives from item $c$ in trip $t$, ignoring all other items in their basket. This can ignore important item to item interaction effects. For example, if a customer has already placed crackers in their basket, then the utility obtained by buying cheese would be greater than if the cheese was purchased without crackers. To model these item to item interactions, SHOPPER introduces the term $\rho_c$.

If we consider two items $c$ and $c'$, then $\rho_c$ represents the interaction effects of product $c$ has with $c'$. Using the item attribute vector, $\alpha_{c'}$ estimated earlier in equation 1 we can estimate the complementary and substitution effects of different products. If $\rho_{c}^T \alpha_{c'}$ is large, then the presence of item $c'$ in the basket increases the benefit choosing item $c$. In the language of economics, we can say these two products are complements. Even though two items may have different underlying latent vectors, i.e. cheese and crackers, they may still be purchased together because they are complements. Conversely, when $\rho_{c}^T \alpha_{c'}$ is smaller, we can interpret these items as substitutes. Even if the latent item vectors are similar, the two products can still be substitutes, i.e. crackers and crispbread (Knäckebrot).

In the SHOPPER model, when the customer is deciding whether to put item $c$ in the basket, they do not consider the item to item interaction of every single item individually. Instead SHOPPER takes the average value of the latent item vectors of the products already in the basket and calculates the inner product with $\rho_{c}$. The scaling factor $ \frac{1}{(i - 1)}$ captures the idea that in larger baskets each individual item has a smaller effect on the addition of new product in the basket.

Additionally, the utility function is additive in the other items in the basket, meaning SHOPPER assumes item interactions within the basket are linear. This is a point that could be studied for future work, to see if item interactions are linear or not.

## Equation 3 & 4 (Utility and Choice Probabilities) [¶](#equation-3--4-utility-and-choice-probabilities)

From equation 1 and 2 we calculate the term $\Psi(c, y_{t, i-1})$, which is interpreted as the utility a customer receives from item $c$, given the week, price, and other items in the basket. The customer's problem then becomes:
$$max \space U_{t,c}(y_{t, i-1}) = \Psi(c, y_{t, i-1}) + \epsilon_{t,c} $$

Those familiar with discrete choice modelling will quickly recognize the above formula as a random utlity model. Upon entering the store, the customer is presented with a set of alternatives they must choose from. The customer chooses the alternative that maximizes their utility, i.e. if item $c$ generates a higher utility than item $c'$, the customer chooses item $c$. In this case $U(c) > U(c')$. The full utility $U$ is known to the customer, but is not fully observable by another person. There is some random component that is not observable by the researcher. In SHOPPER, $\Psi(c, y_{t, i-1})$ represents the deterministic portion of the customer's utility, the portion that can be modelled, and $\epsilon_{t,c}$ represents the random portion of utility. SHOPPER assumes $\epsilon_{t,c}$ has zero mean, is i.i.d and follows a Gumbel distribution.

### Random Utility Model [¶](#random-utility-model)

Let $U_{tc}$ represent the utility a customer receives from choosing item c on trip $t$ and $C$ represent the full set of alternatives ($c = 1, 2, ... ,C$; with $C\geq2$) the customer can choose from.

The customer then chooses item $c$ from the set of alternatives that provides the highest utilities. The customer chooses item c where $U_{tc} > U_{tc'} \; \forall \: c \neq c' $. Therefore the probability that the customer chooses a specific alternative can be written as $Prob(U_{tc} > U_{tc'} \; \forall \: c \neq c')$. Plugging in our utility function from before (equation 3) results in the following expression:

$P_c = p(\Psi(c, y_{t, i-1}) + \epsilon_{t,c} > \Psi(c', y_{t, i-1}) + \epsilon_{t,c'}, \; \forall \: c \neq c') $

Using the derivation from Train 2009 (Discrete Choice Methods WIth Simulation) and the assumption of i.i.d error terms  that follow a Gumbel distribution we can show that the probability of choosing item $c$ can be described using a softmax function.

$p(y_{ti} = c | \mathbf{y}_{t, i-1}) = \frac{exp\{\Psi(c, \mathbf{y}_{t, i-1})\}}{\Sigma_{c' \notin y_{t, i-1}}exp\{\Psi(c', \mathbf{y}_{t, i-1})\} } $

To calculate the probability item $c$ is chosen we exponentiate the customer's utility in trip $t$ with respect to item $c$ and divide that by the sum of all other utilites raised to base $e$. Keen observers will note that this is simply a softmax function over all possible items. This can be difficult to costly to calculate, so an alternate method is used instead (see Appendix)

## Equation 5 (Putting it All Together) [¶](#equation-5-putting-it-all-together)

We have now seen the decision making process of our customer when choosing a single product, but the overall goal of the shopping trip is to maximize the utility of the entire basket. In this case, the shopper only cares about the items that are in their basket at the end of the trip and not necessarily in what order they were placed in.
$$\widetilde{U}_t(\mathcal{Y_t}) = \displaystyle\sum_{c \in \mathcal{Y_t}} \psi_{tc} + \frac{1}{|\mathcal{Y_t}| - 1} \displaystyle\sum_{(c, c') \in \mathcal{Y_t} \times \mathcal{Y_t}:c'\neq c} v_{c',c}$$

From the above equation, we see that the customer chooses items such that the unordered set of items $\mathcal{Y_t}$ maximizes utility. $\displaystyle\sum_{c \in \mathcal{Y_t}} \psi_{tc}$ represents the sum of the utilities earned from each item in the basket. $\frac{1}{|\mathcal{Y_t}| - 1} \displaystyle\sum_{(c, c') \in \mathcal{Y_t} \times \mathcal{Y_t}:c'\neq c} v_{c',c}$ represents the utility gained from item-item interactions in the basket.

## Probabilities of Unordered Baskets [¶](#probabilities-of-unordered-baskets)

Equation 5 from above actually represents the utility a shopper gains from the unordered set of items in their basket, but using our softamx function from earlier, we can calculate the probability of an _ordered_ basket.

The probability of an ordered basket is simply the product of the individual choice probabilities. For example, the probability of getting the basket $(bread, eggs, milk)$ equals $p(bread) \times p(eggs) \times p(milk)$. In general, it can be written as:
$$p(\mathbf{y}_t | \rho, \alpha) = \displaystyle \prod_{i=1}^{n_t} p(y_{ti}| \mathbf{y}_{t, i-1}, \rho, \alpha) $$

where $n_t$ represents the last item in the basket for that trip. However, in most offline real-world datasets, the order in which items are added to the basket is not observed. To account for this, Shopper calculates the likelihood of an unordered set of items $\mathcal{Y}_t$, by summing over all the possible orderings. In our small example above, this corresponds to $$p(\{bread, eggs, milk\}) = p((bread, eggs, milk)) + p((eggs, bread, milk)) + p((eggs, milk, bread)) + ...$$

For only three items, we have to consider six different probabilities. As the basket size grows to $n$ items, we then have to sum over $n!$ probabilities, which can become slow to calculate very quickly. The probability of an unordered basket can be generalized as:
$$ p(\mathcal{Y}_t | \rho, \alpha) = \displaystyle \sum_{\pi}p(\mathbf{y}_{t, \pi}| \rho, \alpha) $$

where $\pi$ is a premutation of the items in the basket. SHOPPEr will generate its latent variables to maximize the log of this probability. The closer the log-likelihood is to zero, the better the latent variables describe the data.

### How is SHOPPER different [¶](#how-is-shopper-different)

As we have seen, SHOPPER combines techniques and knowledge from various domains to create a generative model that can estimate user preferences, seasonal factors, and price elasticities. SHOPPER imposes a structure on the model, noteably using the Random Utility Model as a model of customer behaviour. This model has been in frequent use in marketing since the 1970s. By applying a structure to the model, SHOPPER users are able to draw insights from the results and make counterfactual assumptions using the price elasticity estimates.

## Nitty Gritty [¶](#nitty-gritty)

### Matrix Factorization [¶](#matrix-factorization)

Now that we have seen how SHOPPER works conceptually, we can look into how the latent variables are actually generated.

One option of creating latent variables is to simply create a one-hot encoded matrix. We could create a vector for each shopper that has a 1 for each item they have bought, and a zero for products they did not buy. If we repeat this process for all shoppers, we create our one hot encoded cusomer-item matrix. The rows of our matrix can actually be considered an embedding for that shopper, as we have now converted a shopper's purchases into a numerical vector. Similarly the columns of our matrix can be considered to be the item embedding, as we now see which shoppers purchased that item. The following shows an example of what a one-hot encoded matrix would look like:

![alt text](/blog/static/img/seminar/group3_shopper/shopper_mf.jpg)

We can see that it is easy to summarize across shoppers or products, but it is difficult to see any underlying patterns among users or products. A large number of shoppers and products can also cause memory problems very quickly, the dimensions of the matrix are $number \ of \ unique \ shoppers * number \ of \ unique \ items$. Much of the matrix is also sparse, meaning most entries are 0, because most shoppers do not purchase or even consider all of the unique products for sale. For our small example, this is not a problem but larger datasets with thousands of unique shoppers and thousands of unique products, require a higher memory capacity.

To solve this problem we turn to MF. MF posits that there are a set of latent variables (unobservable variables) that describe each shopper and each item, such that when multiplied together the resulting product is our observed shopper-item matrix. In our small example, we believe that there are 5 latent variables that can accurately describe shoppers and items. Therefore we create two matrices $Shoppers$ and $Items$, that hold our shoppers' latent variables and items' latent variables, respectively. To begin with, we initialize the matrices randomly. They would look something like this, where $l_i$ represents latent variable $i$:

![alt text](https://drive.google.com/uc?id=1kvSUgun_3u0LBpwZV6nIqdkvnRHk3pZn)

![alt text](https://drive.google.com/uc?id=18KNFXDTUyJw57f99IziUOPZGSfu-aMeY)

The rows of our $Shoppers$ matrix represent latent shopper variables,and the columns of our $Items$ matrix represent latent item variables. We need to estimate our latent variables such that $Shoppers \cdot Items $ is equal to our observed shopper-item matrix.
To see if our factorization is going well, we define an error function. In traditional matrix factorization, an error ter such as the matrix norm or mean squared error is used to calculate the error. However, the authors of SHOPPER define a variational inference algorithm that optimizes the factorization based on the log-likliehood of baskets described earlier. The algorithm is out of the scope of this post, but interested readers can see it here.

## Our Task [¶](#our-task)

We will now attempt to replicate the SHOPPER model using a neural network framework. The trick here is to turn SHOPPER, a generative model, into a discriminative model, one that optimizes by making a prediction. We will take a deeper look into the unique challenges of this problem and our corresponding results.

# Data Exploration and Preparation [¶](#data-exploration-and-preparation)

Our dataset contains all purchases made in an unspecified grocery chain made between April 2015 and March 2017 in Germany. The information available to us is the user ID, shopping basket ID, product description, category and sub-category of the product, price and date of purchase.

### Read in data [¶](#read-in-data)

Here we seperate our file into chunks and drop rows that have NA in the article_text and user_id columns. Since we will be estimating individual preferences and item attributes, we require these two columns for the model.
In [0]:

```hl-ipython3
#import pandas library
import pandas as pd
pd.set_option('display.max_rows', 500)
from google.colab import drive
drive.mount('/content/drive')
import datetime

# read csv in chunks
df = pd.read_csv('baskets.gz', sep = '|', chunksize = 1000000)

# read in product information
products = pd.read_table('master_product.tsv.gz')

# identify products that have 'PFAND' in the article text and remove them
# also remove products that have a category name of 'other', ie products like shirts, irons, etc
pfand_prods = products[products['article_text'].str.contains("PFAND")].rpid.to_list()
pfand_prods.extend(products[products['subcategory_name'].str.contains("Pfand")].rpid.to_list())
pfand_prods.extend(products[products['category_name'] == 'other'].rpid.to_list())
pfand_prods = list(set(pfand_prods))

# append each chunk df here
chunk_list = []

# Each chunk is in df format
for chunk in df:
    # remove unnecessary columns
    del chunk['store_id'], chunk['till_id'], chunk['gtin']

    # Preliminary data cleaning
    chunk_filter = chunk.merge(products, how = 'left', left_on ='product_id', right_on = 'rpid')
    chunk_filter = chunk_filter[~chunk_filter['product_id'].isin(pfand_prods)] # remove Pfand and other category items
    chunk_filter = chunk_filter.dropna(subset = ['article_text'], axis = 0, how = 'any') # remove NA's in article_text
    chunk_filter = chunk_filter.dropna(subset = ['user_id'], axis = 0, how = 'any') # remove rows with NA's in user_id

    # drop duplicate items in each basket, we only want to consider if the item was bought with other products or not,
    # not the number of products. In our data, 2 of the same products in a basket are entered as two seperate rows.
    chunk_filter = chunk_filter.drop_duplicates(subset = ['user_id', 'basket_hash', 'article_text'])

    # convert date to datetime
    chunk_filter['day'] = pd.to_datetime(chunk_filter['day'])

    # extract week
    chunk_filter['week'] = chunk_filter['day'].dt.week

    # extract year
    chunk_filter['year'] = chunk_filter['day'].dt.year

    # create bought column equal to 1 for prediction task later
    chunk_filter["bought"] = 1

    # calculate the number of products in each basket
    chunk_filter['prods_in_basket'] = chunk_filter.groupby('basket_hash')['article_text'].transform('count')

    # keep only the columns we need
    chunk_filter = chunk_filter[['basket_hash', 'article_text', 'user_id', 'price', 'day', 'category_name', 'subcategory_name', 'prods_in_basket', 'week', 'year', 'bought']]

    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk_filter)

# concat the list into dataframe
df_concat = pd.concat(chunk_list)
```

In [0]:

```hl-ipython3
df_concat.describe([0.02, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98])
```

Out[0]:

|  | basket_hash | user_id | price | prods_in_basket |
| --- | --- | --- | --- | --- |
| count | 1.205839e+08 | 1.205839e+08 | 1.205839e+08 | 1.205839e+08 |
| mean | 6.689876e+14 | 2.980680e+05 | 1.760459e+00 | 1.220974e+01 |
| std | 5.325177e+18 | 3.287824e+05 | 1.771819e+00 | 9.472519e+00 |
| min | -9.223371e+18 | 1.000000e+00 | -2.495900e+02 | 1.000000e+00 |
| 2% | -8.854942e+18 | 6.487000e+03 | 2.500000e-01 | 2.000000e+00 |
| 10% | -7.377275e+18 | 3.838300e+04 | 5.500000e-01 | 3.000000e+00 |
| 25% | -4.610714e+18 | 1.209930e+05 | 8.900000e-01 | 6.000000e+00 |
| 50% | -4.416611e+14 | 2.085390e+05 | 1.390000e+00 | 1.000000e+01 |
| 75% | 4.612632e+18 | 3.677930e+05 | 1.990000e+00 | 1.600000e+01 |
| 85% | 6.458280e+18 | 4.530980e+05 | 2.690000e+00 | 2.100000e+01 |
| 90% | 7.379322e+18 | 5.011380e+05 | 2.990000e+00 | 2.400000e+01 |
| 95% | 8.302024e+18 | 1.378196e+06 | 4.200000e+00 | 3.100000e+01 |
| 98% | 8.854137e+18 | 1.480803e+06 | 5.990000e+00 | 3.900000e+01 |
| max | 9.223371e+18 | 1.835212e+06 | 9.999900e+02 | 1.610000e+02 |

## Data exploration [¶](#data-exploration)

### Number of products in a basket [¶](#number-of-products-in-a-basket)

To get a sense of our shopper's habits, we first take a look at the distribution of the number of items in each basket. This will also become important later for our modelling.
In [0]:

```hl-ipython3
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)

# histogram of number of products in baskets
sns.distplot(df_concat['prods_in_basket'])

# mean baskets = 12
# median baskets = 10
```

Out[0]:<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f521b209a90&gt;</pre>

We have a large variance in the number of products in each basket, but in order to be able to get the item to item embeddings, we consider baskets that only have more than 2 products in the basket. Also, we do not want baskets that have too many products, as the basket value would overwhelm any of our other embeddings. For this reason we limit the basket size to maximum of 40 products.
In [0]:

```hl-ipython3
#filter out baskets with less than two products and more than 40
df_concat = df_concat[(df_concat['prods_in_basket'] >= 2) & (df_concat['prods_in_basket'] <= 40)]
sns.distplot(df_concat['prods_in_basket'], kde = False, rug=True)
```

Out[0]:<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f5479b346a0&gt;</pre>

In addition to the number of items in a basket, we also consider the number of times a shopper has visited the store. It does not make sense to estimate user preferences when we have such a small number of visits for that user. We would only be adding noise to the model, so we consider dropping infrequent visitors.
In [0]:

```hl-ipython3
#count number of baskets by user ID
no_baskets_by_user = df_concat.groupby('user_id')['basket_hash'].nunique()

print(no_baskets_by_user.describe([0.02, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98]))
sns.distplot(no_baskets_by_user)
```

```
count    561110.000000
mean         33.023765
std          57.186352
min           1.000000
2%            1.000000
10%           1.000000
25%           1.000000
50%          10.000000
75%          40.000000
85%          69.000000
90%          94.000000
95%         141.000000
98%         211.000000
max        1584.000000
Name: basket_hash, dtype: float64
```

Out[0]:<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f00536c86a0&gt;</pre>

We have many users that have only visted the store once in our dataset. It is impossible for us to train and test on users that we don't have enough data for, so we will only consider users that have visted the store more than 10 times.
In [0]:

```hl-ipython3
no_baskets_by_user = no_baskets_by_user[no_baskets_by_user >= 10]

print(no_baskets_by_user.describe([0.02, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98]))
sns.distplot(no_baskets_by_user)
```

```
count    283181.000000
mean         62.769186
std          68.469905
min          10.000000
2%           10.000000
10%          13.000000
25%          20.000000
50%          39.000000
75%          79.000000
85%         112.000000
90%         140.000000
95%         193.000000
98%         271.000000
max        1584.000000
Name: basket_hash, dtype: float64
```

Out[0]:<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f0053005a58&gt;</pre>

In addition to limiting ourselves to shoppers with more than 10 visits, we also consider a sample of shopper IDs. Our total dataset includes about 25 000 unique shoppers, and in the interest of focusing on modelling and experimenting quickly, we consider a subsample of them. Throughout the project we considered sampling 5000, 10 000, and 15 000 shoppers. The goal being to show that our model can generate latent variables for tens of thousands of shoppers. Going forward, we will consider a sample of 15 000 unique shopping IDs.
In [0]:

```hl-ipython3
import numpy as np

# take a random sample of 15000 user IDs
user_sample = np.random.choice(no_baskets_by_user.sort_values(ascending=False).index, 15000)
```

In [0]:

```hl-ipython3
no_baskets_by_user[user_sample].describe()
```

Out[0]:

```
count    15000.000000
mean        61.957200
std         67.123684
min         10.000000
25%         20.000000
50%         39.000000
75%         78.000000
max       1248.000000
Name: basket_hash, dtype: float64
```

In [0]:

```hl-ipython3
# filter our df so only user IDs from our sample are in df
df_filter = df_concat[df_concat['user_id'].isin(user_sample)]
```

After filtering out products with only one item or more than 40 items in the basket, there were 12 items in a basket on average.
Our data is classified into 119 categories (e.g. Fruits) and 911 unique subcategories (e.g. Apples).
In [0]:

```hl-ipython3
# select necessary columns
total_filter = df_filter[['basket_hash', 'article_text', 'user_id', 'price', 'day', 'category_name', 'subcategory_name', 'week', 'year', 'bought']]
```

We now have a dataframe with 15000 unique users, and only necessary columns. Now we extract the week and year from the day column so that we can join our prices table, and use the week column for our week embeddings.
In [0]:

```hl-ipython3
total_filter.head()
```

Out[0]:

|  | basket_hash | article_text | user_id | price | day | category_name | subcategory_name | week | bought |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -5739786015782748519 | Frischli H-MILCH 1,5% FETT 1L | 5758.0 | 0.55 | 2015-04-23 | milk | H-Milch | 17 | 1 |
| 1 | -5739786015782748519 | AMERICAN CLUB FILTERHUELSEN 200ER | 5758.0 | 0.99 | 2015-04-23 | tobacco | Sonstiger Rauchbedarf, z. B. Zigarettenp | 17 | 1 |
| 2 | -5739786015782748519 | JEDEN TAG BABY FEUCHTTÜCHER SENSITIV DUO-PACK ... | 5758.0 | 1.99 | 2015-04-23 | baby_body_care | Pflegetücher für die Säuglings- und Kind | 17 | 1 |
| 3 | -5739786015782748519 | JUETRO BROCCOLI  750G TIEFGEFROREN | 5758.0 | 1.99 | 2015-04-23 | frozen_vegetables | Speisekohl | 17 | 1 |
| 4 | -5739786015782748519 | PALL MALL ALLROUND FULL FLAVOUR 99G | 5758.0 | 15.50 | 2015-04-23 | tobacco | Feinschnitt-Tabak | 17 | 1 |

### Item View [¶](# item-view)

In [0]:

```hl-ipython3
# check item sales by day
total_filter['month'] = total_filter['day'].dt.month
total_filter['dayofweek'] = total_filter['day'].dt.strftime('%a').astype(CategoricalDtype(categories = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], ordered=True))
```

In [0]:

```hl-ipython3
total_filter.describe(include=['object'])
```

Out[0]:

|  | article_text | day | category_name | subcategory_name |
| --- | --- | --- | --- | --- |
| count | 5882684 | 5882684 | 5882684 | 5882684 |
| unique | 28298 | 686 | 110 | 911 |
| top | PRESSERZEUGN.7% MWST | 2015-10-02 | vegetables | Joghurt, H-Joghurt, Joghurtzubereitungen |
| freq | 77613 | 22524 | 479195 | 176391 |

#### Bestsellers [¶](# bestsellers)

Most purchased items are:

1. Newspapers
2. Pfand (Bottle deposit)
3. Bananas
4. Plastic bag
5. Bread rolls
6. Bigger plastic bag
7. Pickles
8. Bananas (bio)
9. Tomatoes
10. Butter
11. Cherry tomatoes
12. Potatoes
13. Carrots
14. Whole milk
15. Skinned milk

(based on 5K)
In [0]:

```hl-ipython3
# check highest selling items
df_top_items = total_filter.groupby(['article_text'])['article_text'].agg(
    {"purchases": len}).sort_values(
    "purchases", ascending=False).head(20).reset_index()

df_top_items
```

```
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:2: FutureWarning: using a dict on a Series for aggregation
is deprecated and will be removed in a future version. Use                 named aggregation instead.

    >>> grouper.agg(name_1=func_1, name_2=func_2)
```

Out[0]:

|  | article_text | purchases |
| --- | --- | --- |
| 0 | PRESSERZEUGN.7% MWST | 77613 |
| 1 | Bananen Cavendish KG | 57026 |
| 2 | A&P Schrippe Discount-Konzept 60g, Prebake | 37711 |
| 3 | Schlaufentraget. Kaisers Im green | 36252 |
| 4 | Sinustragetasche Kaiser`s Im green | 34873 |
| 5 | Gurken 350g-400g - NL | 30864 |
| 6 | Bio Bananen KG | 26237 |
| 7 | Tomaten Rispen  KG | 24261 |
| 8 | Deutsche Markenbutter Gold mildges. 250g | 24193 |
| 9 | Tomaten Cocktailstrauch 300g Schale | 20284 |
| 10 | Frischgold Vollmilch 3,5% 1 Ltr. ESL | 18905 |
| 11 | Speisekartoffeln KG v.f.k. | 18764 |
| 12 | Frischgold Milch 1,5% 1Ltr.  ESL | 18659 |
| 13 |  "Äpfel Braeburn KG /"mild säuerlich/" " | 18637 |
| 14 | Speisemoehren / Karotten  KG gelegt | 18399 |
| 15 | Frischli H-MILCH 1,5% FETT 1L | 17024 |
| 16 | NATURKIND Bio Eier 6er | 16291 |
| 17 | EIER AUS BODEN-       HALTUNG KL.M 10ER | 16173 |
| 18 | Paprika Rot  KG | 15162 |
| 19 | Schrippe (BS)  ca. 60g, Discount Konzept | 14300 |

In [0]:

```hl-ipython3
# because shopping bags are one of the most frequently purchased products and they do not hold any prediction value, we exclude them from our model
total_filter = total_filter.drop(total_filter[total_filter.subcategory_name == "Tragetaschen"].index)
```

#### Price Points [¶](#price-points)

We can observe that most products are selling at low price points - most of the store's revenue comes from products cheaper than 5 Euros.
This is specific to grocery stores.
In [0]:

```hl-ipython3
n, bins, patches = plt.hist(df['price'], bins=[0,1,2,3,4,5,7.5,10,15,21], facecolor='blue', alpha=0.5)
plt.title("Sales by price points", fontsize=15)
plt.xlabel("Price in Euros")
plt.ylabel("Products bought")
plt.show()

# most sales comes from products <2 Euros
```

In [0]:

```hl-ipython3
n, bins, patches = plt.hist(df['price'], bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5], facecolor='blue', alpha=0.5)
plt.title("Sales by price in low price points", fontsize=15)
plt.xlabel("Price in Euros")
plt.ylabel("Products bought")
plt.show()
```

#### Seasonality [¶](#seasonality)

In [0]:

```hl-ipython3
df_by_month = total_filter[total_filter['year']==2016].groupby('month').agg(
    {'basket_hash': pd.Series.nunique, #number of baskets
    'user_id': pd.Series.nunique, #customers
    'article_text': pd.Series.count, #items bought
    'price': np.sum #revenue
    }
)

df_by_month.rename(columns={'basket_hash':'shopping_trips',
                          'user_id':'unique_customers',
                          'article_text':'items_bought',
                          'price':'revenue'},
                 inplace=True)
df_by_month['unique_customers'] = df_by_month['unique_customers'].astype(int)
df_by_month['revenue'] = df_by_month['revenue'].round(2)
df_by_month
```

Out[0]:

| <br>month | shopping_trips<br> | unique_customers<br> | items_bought<br> | revenue<br> | month<br> |
| --- | --- | --- | --- | --- | --- |
| 1 | 45401 | 9134 | 301713 | 508834.75 | 1 |
| 2 | 45108 | 9163 | 289596 | 499493.49 | 2 |
| 3 | 43167 | 8990 | 280099 | 497737.85 | 3 |
| 4 | 44900 | 8937 | 291258 | 515948.13 | 4 |
| 5 | 41562 | 8708 | 269960 | 486341.13 | 5 |
| 6 | 42821 | 8592 | 273018 | 480978.99 | 6 |
| 7 | 41606 | 8468 | 267830 | 471895.04 | 7 |
| 8 | 39848 | 8083 | 252980 | 433307.37 | 8 |
| 9 | 39378 | 8053 | 254060 | 435413.03 | 9 |
| 10 | 36095 | 7814 | 242590 | 422013.15 | 10 |
| 11 | 36403 | 7687 | 242495 | 426875.51 | 11 |
| 12 | 36017 | 7491 | 248327 | 467408.78 | 12 |

In [0]:

```hl-ipython3
plt.bar(df_by_month['month'],df_by_month['revenue']/1000)
plt.ylabel('Monthly sales in Euros (thousands)')
plt.title('Revenue per Month', fontsize=15)
plt.show()
# Most frequent month: March
# Least frequent month: October
# in 2016 only
```

In [0]:

```hl-ipython3
plt.bar(df_by_weekday['dayofweek'],df_by_weekday['revenue']/1000)
plt.ylabel('Total sales in Euros (thousands)')
plt.title('Revenue by Day of week', fontsize=15)
plt.show()
# shops are closed on Sunday!
```

In [0]:

```hl-ipython3
# Which types of products (sub-categories) have most variation in sales between different months?

df_subcat_by_month = total_filter.groupby(['subcategory_name','month'])['article_text'].transform('count')
df_subcat_by_month.rename(columns={'article_text':'items_bought'}, inplace=True)
df_subcat_by_month_range = pd.DataFrame( ((df_subcat_by_month['items_bought'].max(level=0) - df_subcat_by_month['items_bought'].min(level=0)) / df_subcat_by_month['items_bought'].mean(level=0) ), )
df_subcat_by_month2 = df_subcat_by_month_range.sort_values(by='items_bought',ascending=False).reset_index()
df_subcat_by_month2.rename(columns={'items_bought':'relative_change_between_months'},inplace=True)
df_subcat_by_month2.head(20)
# relative_change_between_months = 4 means that the difference between the best-selling and worst-selling month for the category is 4 times higher than the mean sales
```

Out[0]:

|  | subcategory_name | relative_change_between_months |
| --- | --- | --- |
| 0 | Oster-Marzipan | 9.312500 |
| 1 | Weihnachtshohlfiguren, -schokolade, -gel | 7.566879 |
| 2 | Feigen | 5.931148 |
| 3 | Speisekohl, Blatt- und Stängelgemüse | 5.786458 |
| 4 | Kanditen und Belegfrüchte, z. B. solche | 4.962963 |
| 5 | Wildfleisch, roh (ohne Wildgeflügel) | 4.938776 |
| 6 | Sonstige Brühwürstchen | 4.928571 |
| 7 | Wildgeflügel | 4.918033 |
| 8 | Oster-Hohlfiguren, -pralinen, -gelee, -f | 4.855491 |
| 9 | Hasel-, Para-, Pekan- und Walnüsse, Nuss | 4.639405 |
| 10 | Spargel | 4.539398 |
| 11 | Enten | 4.486842 |
| 12 | Erd- und Paranusskerne, Cashewkerne | 4.281407 |
| 13 | Weizenmischbrot ganz BD | 4.260870 |
| 14 | Sonstige Saisonartikel aus Schokolade, M | 4.002848 |
| 15 | Gänse | 4.000000 |
| 16 | Wildgeflügel Selbstbedienung | 3.836066 |
| 17 | Spezialbrot ganz BD | 3.826087 |
| 18 | Obst- und Gemüsesäfte | 3.750000 |
| 19 | Suppen, trocken oder pastös (einschl. In | 3.693694 |

## Data preparation [¶](#data-preparation)

As we have seen, the bestselling products include the "Pfand" (the deposit for bottles and cans) and shopping bags. These do not have much value for our predictive model, as in most cases the prices for these are set by law and hence coupons are not applicable, therefore we remove these two product subcategories.
Apart from the specific products, we have made a few assumptions about general shopping trends. Firstly, assuming that good personal recommendations can be made only for loyal customers, we have restricted our dataset to customers who have visited the store at least 10 times. Secondly, to be able to utilize the features of other products in the basket, we allow only baskets with at least two items, and at most 40 items to avoid outliers. Thirdly, we predict only which items the customer is going to consider, no matter the quantity, therefore we disregard all duplicate entries (originally, if a customer bought three bananas, we would have three identical rows in our dataset).

Since we are turning a generative model into one that makes predictions, we need to make adjustments to our dateset. SHOPPER fits the latent variables it generates to the shopping baskets data and minimizes the log likelihood. However, in order for our model to work we must give it a prediction task on which to optimize.

One problem that occurs when facing such a task is that in a dataset with actual shopping data, there will be only products that were bought by customers and not products that were considered but not bought. However, when building a model that is designed to predict what product a customer is considering to buy it should also train on products that were not bought, to correctly generate the preferences of a shopper.

A solution for this problem is to add products that are not bought into the data so that the model sees both, products that were bought and products that were not bought. Such a process needs some assumptions and factor in specifics of shopping in order to sample new products into the data.

The "most correct" way for us to model shopper choices would be for us to treat the problem as a discrete choice problem. We would have to calculate the probability that a specific item was chosen by the shopper over all available products. This would essentially become a softmax function over 35 000 items in our case. Since this is not technically feasible, we use a trick inspired by word2vec.

Instead of considering one item against all other items, we instead sample items that the shopper did not place in their basket, assign them a "bought" value of 0, and predict whether or not the shopper bought the item. The problem then becomes a binary choice problem, "bought" or "not bought", over bought items and our sampled not-bought items. This speeds up our calculations and should also generate robust results, as seen in other applications (source).

For our sampling procedure, we make the following assumptions:

1. A shopper considers products from the same subcategories as the items in his/her basket. For example, if the customer buys spaghetti he also considered penne or maybe spaghetti from a different brand on the shelf.
2. The price of the product that was not bought will be equal to the most frequent price the particular product was bought at in the respective week and year

These assumptions lead to a certain expectation about the shopping habits of customers. We assume that if a shopper has bought a product, then he must have at least seen other products from the same sub-category and chosen not to purchase them. This assumption leaves room for optimisation, as it is debatable if the customer only considers one other product for every product he bought. In general, customers most likely consider more than one item when shopping, however then the model would rely on even more generated data which we wanted to prevent. The suitability number of not-bought product sampled can be discussed and for further research needs to be addressed again, nevertheless we move forward.

Below we describe our sampling procedure for not-bought items:

1. Group all products by subcategory name, year, and week.
2. Group shopper purchases into baskets, by grouping on user id and       basket hash.
3. Randomly sample an item from the full group that is in the same
subcategory as the item in the shopper's basket, and not already
in the shopping basket.
4. Fill in the rows of the new products with data of the remaining
columns from the original product except for price.
5. Calculate the most frequent price at which a particular product
was sold in a particular week and year. For prices that are        unavailable we use the previous week's price, and if still unavailable, we use the following week's price.
6. Mark sampled products as not bought, by assigning a bought value of

We first create groups for all products which were available in a subcategory for each week and year combination. We will use these groups to sample unbought products into the dataset which we are using for our model.
In [0]:

```hl-ipython3
# Step 1
# create groups of products by subcategory, week and year
groups = df_concat.groupby(['year','week','subcategory_name']).apply(lambda x: x['article_text'].unique())
groups.head()
```

Out[0]:

```
year  week  subcategory_name
2015  16    Ausländische Qualitätsschaumweine (Sekt)           [prosecco frizz. doc bella aura oro 0.75l]
            Bananen                                                                [bananen cavendish kg]
            Bierschinken Selbstbedienung                                              [bierschinken 200g]
            Blatt- und Stängelgemüse (ohne Salate)                    [iglo port.blattspinat 500g tiefge]
            Brötchen BD                                 [meisterstück / meisterschrippe (bs), a&p schr...
dtype: object
```

Next we group the products a user bought in one shopping trip into a basket, which contains all the products he bought in this trip.
In [0]:

```hl-ipython3
# Step 2
# create baskets for each user containing the products of a particular shopping trip
baskets = total_filter.groupby(['user_id','basket_hash']).apply(lambda x: x['article_text'].unique())
baskets
```

Out[0]:

```
user_id    basket_hash
442.0      -8717244567711960099    [haribo konfekt  lakritz  200g, waldquelle cla...
           -8706933430938556035    [lorenz erdnuss locken jumbos 225g, rufin fein...
           -8685103104264859614    [bio zentrale ahornsirup 250ml, frischgold mil...
           -8680938939970530513                 [burger knäckebrot delikatesse 250g]
           -8584861831361893801    [frischgold milch 1.5% 1ltr.  esl, schrippe (b...
                                                         ...
1821754.0   6460538675951702206    [schrippe /  spitzbrötchen premium (bs)  ca 75...
            7798499040000251981    [speiskartoffeln 2.5 kg m.k. sorte: siehe etik...
            7919008002285420495             [bauernbrötchen nach mailänder art (bs)]
            8712853537921653904    [blutorangen sanguinelli 1kg netz, geramont cr...
            8766703745985938372    [baerenmarke milch 1.8% fett 1 liter, mandarin...
Length: 305890, dtype: object
```

We then take the difference between the group of all products and the basket of products of a certain shopping trip. From this difference, we randomly select one product from the same subcategory as a product that was bought. Thereby creating a new product which was not bought by this customer in his shopping trip.
In [0]:

```hl-ipython3
# Step 3
# randomly taking a product from the same subcategory as a product that was bought

import random
new_rows = pd.Series([random.choice(list(set(groups[(x.year, x.week, x.subcategory_name)]).difference(set(baskets[(x.user_id, x.basket_hash)])))) for x in total_filter.itertuples() if len(list(set(groups[(x.year, x.week, x.subcategory_name)]).difference(set(baskets[(x.user_id, x.basket_hash)])))) > 0])
new_rows
```

Out[0]:

```
0                   baerenmarke haltb.    alpenmilch 1.5% 1l
1                      zwaar blister papier  4x50 blaettchen
2                                 bobini feuchtetuecher 20+4
3                        juetro rahm blum.kohl 300g tiefgefr
4                                      apollo zig.tabak 200g
                                 ...
1152177                                 m&m¿s peanut ei 250g
1152178                           lindt goldhase weiss  100g
1152179               grünländer scheiben gartenkräuter 150g
1152180    gala nr. 1 fein gemahlen 500g vaku  (alte beze...
1152181                café royal lungo forte 10 kapseln=55g
Length: 1152182, dtype: object
```

Next we fill the rows of the new product with data of the remaining columns from the original bought item except for price.
In [0]:

```hl-ipython3
# Step 4
# fill the rows of the new products with data of the remaining columns
# from the original product except for price
new_sample = pd.DataFrame({'basket_hash': [x.basket_hash for x in total_filter.itertuples() if len(list(set(groups[(x.year, x.week, x.subcategory_name)]).difference(set(baskets[(x.user_id, x.basket_hash)])))) > 0],
                           'article_text': new_rows,
                           'user_id': [x.user_id for x in total_filter.itertuples() if len(list(set(groups[(x.year, x.week, x.subcategory_name)]).difference(set(baskets[(x.user_id, x.basket_hash)])))) > 0],
                           'week': [x.week for x in total_filter.itertuples() if len(list(set(groups[(x.year, x.week, x.subcategory_name)]).difference(set(baskets[(x.user_id, x.basket_hash)])))) > 0],
                           'year': [x.year for x in total_filter.itertuples() if len(list(set(groups[(x.year, x.week, x.subcategory_name)]).difference(set(baskets[(x.user_id, x.basket_hash)])))) > 0],
                           'category_name': [x.category_name for x in total_filter.itertuples() if len(list(set(groups[(x.year, x.week, x.subcategory_name)]).difference(set(baskets[(x.user_id, x.basket_hash)])))) > 0],
                           'subcategory_name': [x.subcategory_name for x in total_filter.itertuples() if len(list(set(groups[(x.year, x.week, x.subcategory_name)]).difference(set(baskets[(x.user_id, x.basket_hash)])))) > 0]})

new_sample
```

Out[0]:

|  | basket_hash | article_text | user_id | week | year | category_name | subcategory_name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -5739786015782748519 | baerenmarke haltb.    alpenmilch 1.5% 1l | 5758.0 | 17 | 2015 | milk | H-Milch |
| 1 | -5739786015782748519 | zwaar blister papier  4x50 blaettchen | 5758.0 | 17 | 2015 | tobacco | Sonstiger Rauchbedarf, z. B. Zigarettenp |
| 2 | -5739786015782748519 | bobini feuchtetuecher 20+4 | 5758.0 | 17 | 2015 | baby_body_care | Pflegetücher für die Säuglings- und Kind |
| 3 | -5739786015782748519 | juetro rahm blum.kohl 300g tiefgefr | 5758.0 | 17 | 2015 | frozen_vegetables | Speisekohl |
| 4 | -5739786015782748519 | apollo zig.tabak 200g | 5758.0 | 17 | 2015 | tobacco | Feinschnitt-Tabak |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1152177 | -6536134241320279174 | m&m¿s peanut ei 250g | 122212.0 | 12 | 2017 | seasonal_sweets | Oster-Marzipan |
| 1152178 | -6536134241320279174 | lindt goldhase weiss  100g | 122212.0 | 12 | 2017 | seasonal_sweets | Oster-Marzipan |
| 1152179 | -6536134241320279174 | grünländer scheiben gartenkräuter 150g | 122212.0 | 12 | 2017 | cheese | Hart-,Schnitt-&halbfester Schnittkäse SB |
| 1152180 | -6536134241320279174 | gala nr. 1 fein gemahlen 500g vaku  (alte beze... | 122212.0 | 12 | 2017 | coffee | Röstkaffee, koffeinhaltig, gemahlen |
| 1152181 | -6536134241320279174 | café royal lungo forte 10 kapseln=55g | 122212.0 | 12 | 2017 | coffee | Röstkaffee, koffeinhaltig, gemahlen |

1152182 rows × 7 columns

The only value that is still missing is the price of the product. In our data there are lots of different prices for the same products, depending on season, coupons or other outside factors. Similar to the groups of products we create a list where we can find the most frequent price at which a particular product was sold in a respective week and year. As there can still be a week where we do not have a price for a certain product we use forward and backward fill. Therefore we will always get a price for our new products.

There can be instances where prices are higher or lower depending on the day in a certain week. In the data there are sometimes differences in price even on the same day. However, this sampling approach generates reliable data that can be seen as the majority price of a certain product at which the customer did not buy the product.

Imputing the prices was a necessary step for us, since we did not have a full list of product prices to simply look up from. We only had the prices for sold products. However, grocery stores should have a price list fo rall products available, so this step would not be necessary and could be skipped.
In [0]:

```hl-ipython3
df_prices = df_concat[['article_text', 'price', 'day']]
df_prices['day'] = pd.to_datetime(df_prices['day'])
df_prices['week'] = df_prices['day'].dt.week
df_prices['year'] = df_prices['day'].dt.year
```

In [0]:

```hl-ipython3
def top_value_count(x):
    return x.value_counts().idxmax()
```

In [0]:

```hl-ipython3
# Step 5
# calculate the most frequent price at which a particular product
# was sold in a respective week and year
prices_top_freq = df_prices.groupby(['year','week', 'article_text'])['price']
prices = prices_top_freq.apply(top_value_count).reset_index()
```

In [0]:

```hl-ipython3
prices
```

Out[0]:

|  | year | week | article_text | price |
| --- | --- | --- | --- | --- |
| 0 | 2015 | 16 |  "äpfel braeburn kg /"mild säuerlich/" " | 1.50 |
| 1 | 2015 | 16 |  "äpfel jonagored 4er schale /"aromatisch süss/" " | 0.79 |
| 2 | 2015 | 16 | 1688 steinofenbrot 250g | 1.19 |
| 3 | 2015 | 16 | 6er ostereier aus bodenhaltung regenb. kl. m | 1.49 |
| 4 | 2015 | 16 | a&p bauernbrötchen mit roggen  discount-konzept | 0.50 |
| ... | ... | ... | ... | ... |
| 624860 | 2017 | 13 | werder tomaten ketchup zuckerfrei 450ml | 1.59 |
| 624861 | 2017 | 13 | whiskas junior kalb&  gefluegel 100g sc | 0.55 |
| 624862 | 2017 | 13 | xox knabberkrusten    50g  umstellung ld-sd  z... | 0.99 |
| 624863 | 2017 | 13 | zewa ultra soft 4 lg 6x150 bl | 0.00 |
| 624864 | 2017 | 13 |  äpfel elstar kg | 1.86 |

624865 rows × 4 columns
In [0]:

```hl-ipython3
# Step 6
# add the prices for our new products by merging with the most frequent prices
new_sample2 = pd.merge(new_sample, prices, how = 'left', on = ['year', 'week', 'article_text'])

new_sample2['price'] = new_sample2.groupby('article_text')['price'].transform(lambda x: x.fillna(method = 'ffill'))
new_sample2['price'] = new_sample2.groupby('article_text')['price'].transform(lambda x: x.fillna(method = 'bfill'))

new_sample2
```

Out[0]:

|  | basket_hash | article_text | user_id | week | year | category_name | subcategory_name | price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -5739786015782748519 | baerenmarke haltb.    alpenmilch 1.5% 1l | 5758.0 | 17 | 2015 | milk | H-Milch | 1.19 |
| 1 | -5739786015782748519 | zwaar blister papier  4x50 blaettchen | 5758.0 | 17 | 2015 | tobacco | Sonstiger Rauchbedarf, z. B. Zigarettenp | 0.89 |
| 2 | -5739786015782748519 | bobini feuchtetuecher 20+4 | 5758.0 | 17 | 2015 | baby_body_care | Pflegetücher für die Säuglings- und Kind | 0.99 |
| 3 | -5739786015782748519 | juetro rahm blum.kohl 300g tiefgefr | 5758.0 | 17 | 2015 | frozen_vegetables | Speisekohl | 0.99 |
| 4 | -5739786015782748519 | apollo zig.tabak 200g | 5758.0 | 17 | 2015 | tobacco | Feinschnitt-Tabak | 21.95 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 1152177 | -6536134241320279174 | m&m¿s peanut ei 250g | 122212.0 | 12 | 2017 | seasonal_sweets | Oster-Marzipan | 5.49 |
| 1152178 | -6536134241320279174 | lindt goldhase weiss  100g | 122212.0 | 12 | 2017 | seasonal_sweets | Oster-Marzipan | 2.99 |
| 1152179 | -6536134241320279174 | grünländer scheiben gartenkräuter 150g | 122212.0 | 12 | 2017 | cheese | Hart-,Schnitt-&halbfester Schnittkäse SB | 1.49 |
| 1152180 | -6536134241320279174 | gala nr. 1 fein gemahlen 500g vaku  (alte beze... | 122212.0 | 12 | 2017 | coffee | Röstkaffee, koffeinhaltig, gemahlen | 5.49 |
| 1152181 | -6536134241320279174 | café royal lungo forte 10 kapseln=55g | 122212.0 | 12 | 2017 | coffee | Röstkaffee, koffeinhaltig, gemahlen | 2.89 |

1152182 rows × 8 columns

These sampled product now just have to be marked that they are not bought items in comparison to the original items that were bought and then put together with the original data to create our new dataset.
In [0]:

```hl-ipython3
# products are sampled to add to data and were not bought
new_sample2['bought'] = 0
```

In [0]:

```hl-ipython3
new_sample2 = new_sample2[['basket_hash', 'article_text', 'user_id', 'price', 'category_name','subcategory_name', 'bought', 'week', 'year']]
```

In [0]:

```hl-ipython3
new_sample2
```

Out[0]:

|  | basket_hash | article_text | user_id | price | category_name | subcategory_name | bought | week | year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -5739786015782748519 | baerenmarke haltb.    alpenmilch 1.5% 1l | 5758.0 | 1.19 | milk | H-Milch | 0 | 17 | 2015 |
| 1 | -5739786015782748519 | zwaar blister papier  4x50 blaettchen | 5758.0 | 0.89 | tobacco | Sonstiger Rauchbedarf, z. B. Zigarettenp | 0 | 17 | 2015 |
| 2 | -5739786015782748519 | bobini feuchtetuecher 20+4 | 5758.0 | 0.99 | baby_body_care | Pflegetücher für die Säuglings- und Kind | 0 | 17 | 2015 |
| 3 | -5739786015782748519 | juetro rahm blum.kohl 300g tiefgefr | 5758.0 | 0.99 | frozen_vegetables | Speisekohl | 0 | 17 | 2015 |
| 4 | -5739786015782748519 | apollo zig.tabak 200g | 5758.0 | 21.95 | tobacco | Feinschnitt-Tabak | 0 | 17 | 2015 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 1152177 | -6536134241320279174 | m&m¿s peanut ei 250g | 122212.0 | 5.49 | seasonal_sweets | Oster-Marzipan | 0 | 12 | 2017 |
| 1152178 | -6536134241320279174 | lindt goldhase weiss  100g | 122212.0 | 2.99 | seasonal_sweets | Oster-Marzipan | 0 | 12 | 2017 |
| 1152179 | -6536134241320279174 | grünländer scheiben gartenkräuter 150g | 122212.0 | 1.49 | cheese | Hart-,Schnitt-&halbfester Schnittkäse SB | 0 | 12 | 2017 |
| 1152180 | -6536134241320279174 | gala nr. 1 fein gemahlen 500g vaku  (alte beze... | 122212.0 | 5.49 | coffee | Röstkaffee, koffeinhaltig, gemahlen | 0 | 12 | 2017 |
| 1152181 | -6536134241320279174 | café royal lungo forte 10 kapseln=55g | 122212.0 | 2.89 | coffee | Röstkaffee, koffeinhaltig, gemahlen | 0 | 12 | 2017 |

1152182 rows × 9 columns

We then concatenate the sampled not-bought products and bought products into one dataframe, reseting the index so that they are distributed within the bought products and are not appended in one big chunk e.g at the end.
In [0]:

```hl-ipython3
# putting bought and sampled not bought products into one dataframe
final_df = total_filter.append(new_sample2).sort_index().reset_index(drop=True)
```

```
/usr/local/lib/python3.6/dist-packages/pandas/core/frame.py:7138: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
of pandas will change to not sort by default.

To accept the future behavior, pass 'sort=False'.

To retain the current behavior and silence the warning, pass 'sort=True'.

  sort=sort,
```

In [0]:

```hl-ipython3
final_df
```

Out[0]:

|  | article_text | basket_hash | bought | category_name | price | subcategory_name | user_id | week | year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | frischli h-milch 1.5% fett 1l | -5739786015782748519 | 1 | milk | 0.55 | H-Milch | 5758.0 | 17 | 2015 |
| 1 | baerenmarke haltb.    alpenmilch 1.5% 1l | -5739786015782748519 | 0 | milk | 1.19 | H-Milch | 5758.0 | 17 | 2015 |
| 2 | american club filterhuelsen 200er | -5739786015782748519 | 1 | tobacco | 0.99 | Sonstiger Rauchbedarf, z. B. Zigarettenp | 5758.0 | 17 | 2015 |
| 3 | zwaar blister papier  4x50 blaettchen | -5739786015782748519 | 0 | tobacco | 0.89 | Sonstiger Rauchbedarf, z. B. Zigarettenp | 5758.0 | 17 | 2015 |
| 4 | jeden tag baby feuchttücher sensitiv duo-pack ... | -5739786015782748519 | 1 | baby_body_care | 1.99 | Pflegetücher für die Säuglings- und Kind | 5758.0 | 17 | 2015 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 2315704 | lindt doppelmilch osterfreunde 100g | -6536134241320279174 | 1 | seasonal_sweets | 3.79 | Oster-Marzipan | 122212.0 | 12 | 2017 |
| 2315705 | lindt goldhasen in tasche  100g oster freuden | -6536134241320279174 | 1 | seasonal_sweets | 4.99 | Oster-Marzipan | 122212.0 | 12 | 2017 |
| 2315706 | arla esrom scheiben   45% fett i.tr.175g | -6536134241320279174 | 1 | cheese | 1.89 | Hart-,Schnitt-&halbfester Schnittkäse SB | 122212.0 | 12 | 2017 |
| 2315707 | nescafe dolce gusto cappuccino 8er=200 g | -6536134241320279174 | 1 | coffee | 4.79 | Röstkaffee, koffeinhaltig, gemahlen | 122212.0 | 12 | 2017 |
| 2315708 | nescafe dolce gusto cappuccino 8er=200 g | -6536134241320279174 | 1 | coffee | 4.79 | Röstkaffee, koffeinhaltig, gemahlen | 122212.0 | 12 | 2017 |

2315709 rows × 9 columns

Now that we have our bought and not bought products, we need to add one additional column. We are also estimating item-item interactions so we include an additional column that includes a list of all the other items that were purchased in by the shopper for that specific trip.
In [0]:

```hl-ipython3
# step 7 add other items from basket into seperate column as a list
final_df['other_basket_prods'] = pd.Series([list(set(baskets[(x.user_id, x.basket_hash)]).difference(x.article_text)) for x in final_df.itertuples() ])
```

Currently, we are assuming that the shopper's basket is full and the item we are predicting on is their final, or "checkout" item. We do this because we do not observe the order in which the customer placed items into his basket. In a perfect world, we would have the order items were placed into the basket available to us and we could then iteratively fill the basket. This would give us a better estimation of item-item interactions. Now let's look at modelling.

# Model [¶](#model)

As seen in SHOPPER our approach is to build a model for personalized shopping recommendations. The difference compared to SHOPPER is that we want to translate the whole process they are doing in SHOPPER into the tensorflow structure and see if that can yield the desired results as well.
In order to be as close as possible to the original approach we looked at every part they were doing and translated it step-by-step into our model architecture. Most of the maths behind the shopper paper can be found in our approach as well. This can be seen for example with the way the embeddings are calculated.
In the model architecture we will pick that thought up again and see what formula from shopper goes where in our model architecture.

For this attempt we are taking different inputs from the data and then creating embeddings for these inputs. The embeddings can be seen as latent vector representations, and the dot product of two of these embeddings then gives us the combination contribution to the overall utility. The dot product as well as some more embeddings and the price multiplication then add up to the function of the utility and is put through a sigmoid function in the last dense layer where we receive our probabilities of purchase.

In the model we have these five inputs from our data: user, basket, item, week and price. In the following we will go through the steps that we computed for creating the embeddings, moving on to the dot products and in the end adding it all up into the whole model.

The formula from shopper that we want to use for our neural network is the equation 1 and 2 with latent variables and item-item interactions:
$$\psi_{tc} = \lambda_{c} + \theta_{ut}^T \alpha_{c} - \gamma_{ut}^T \beta_{c} log(r_{tc}) + \delta_{wt}^T \mu_{c} $$$$\Psi(c, y_{t, i-1}) = \psi_{tc} + \rho_{c}^T(\frac{1}{i-1} \displaystyle\sum_{j=1}^{i-1} \alpha_{y_{tj}} )$$

Similar to what we described in the beginning, when we explained the different parts of this formula we will refresh the meaning of the different steps and then show how they are implemented into our model step by step.
The second equation above represents the utility a customer gets from item $c$ in trip $t$, which is what we want to have for our model, because it allows us to predict the probablilty at which a customer will by a certain product. The higher this utility is for an item $c$ the higher the chance the customer will buy it. These utilities change depending on the items in his basket.

## Model architecture [¶](#model-architecture)

Prior to modelling, we must label encode all of our categorical variables, in order to use them as inputs into our embedding layers.
In [0]:

```hl-ipython3
# creating label encoders for items, users and weeks
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(final_df['article_text'])
final_df['encoded_prods'] = le.transform(final_df['article_text'])
final_df['other_basket_prods_encoded'] = final_df['other_basket_prods'].apply(lambda x : le.transform(x))

le_user = LabelEncoder()
le_user.fit(final_df['user_id'])
final_df['encoded_user'] = le_user.transform(final_df['user_id'])

le_week = LabelEncoder()
le_week.fit(final_df['week'])
final_df['encoded_week'] = le_week.transform(final_df['week'])
```

We split our data into 80% training and and 20% test sets. We elect to use a random split for this problem.
In [0]:

```hl-ipython3
# splitting the data into train and test
from sklearn import model_selection

X = final_df.drop(["bought", 'basket_hash', 'category_name', 'subcategory_name'], axis = 1)
Y = final_df["bought"]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size = 0.2, random_state = 42)
```

Here we bring all of our basket sizes to equal length. Recall, that we limited the maximum size of our baskets to 40 products.
In [0]:

```hl-ipython3
from keras.preprocessing.sequence import pad_sequences

largest_basket = X_train['other_basket_prods_encoded'].apply(lambda x: len(x)).max()
basket_prods_train_pad = pad_sequences(X_train['other_basket_prods_encoded'], maxlen = largest_basket + 1, padding = 'post')
basket_prods_test_pad = pad_sequences(X_test['other_basket_prods_encoded'], maxlen = largest_basket + 1, padding = 'post')

basket_prods_train_pad
```

```
Using TensorFlow backend.
```

The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>

We recommend you [upgrade](https://www.tensorflow.org/guide/migrate) now
or ensure your notebook will continue to use TensorFlow 1.x via the `%tensorflow_version 1.x` magic:
[more info](https://colab.research.google.com/notebooks/tensorflow_version.ipynb).
Out[0]:

```
array([[10304, 11712, 30407, ...,     0,     0,     0],
       [ 1377, 32514,  2211, ...,     0,     0,     0],
       [ 2629,   298, 17131, ...,     0,     0,     0],
       ...,
       [ 3840, 23236,  2821, ...,     0,     0,     0],
       [20997, 30855, 35530, ...,     0,     0,     0],
       [ 4165,  3975,  7243, ...,     0,     0,     0]], dtype=int32)
```

In [0]:

```hl-ipython3
import keras
from keras.layers import Input, Embedding, Dot, Reshape, Dense, concatenate, multiply, average, add, Average, Dropout
from keras.models import Model
from keras.optimizers import Adam
```

```
Using TensorFlow backend.
```

The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>

We recommend you [upgrade](https://www.tensorflow.org/guide/migrate) now
or ensure your notebook will continue to use TensorFlow 1.x via the `%tensorflow_version 1.x` magic:
[more info](https://colab.research.google.com/notebooks/tensorflow_version.ipynb).

Here we define our input variables. We will have five inputs: the shopper ID, the item, the price of the item, the week in which the shopping trip took place, and the other items in the basket. All inputs have a shape of 1, aside from the basket input. This is because baskets can be of varying length, so we leave the dimensions as None, in order to be able to accomodate this.
In [0]:

```hl-ipython3
# defining the inputs for our model user, item, price and week
embedding_size = 100
user_len = len(le_user.classes_) + 1
item_len = len(le.classes_) + 1
week_len = len(le_week.classes_) + 1

user = Input(name = 'user', shape = (1,))
item = Input(name = 'item', shape = (1,))
price = Input(name = 'price', shape = (1,))
week = Input(name = 'week', shape = (1,))
basket = Input(name = 'basket', shape = (None,))
```

```
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:541: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
```

The _Item Popularity_,  $\lambda_{c}$, captures the overall item popularity and will be represented in our model by the item popularity embedding that goes straight into our last add function. It has an embedding dimension of 1.
In [0]:

```hl-ipython3
# creating the first embedding layer for item popularity with embedding size of 1
item_pop = Embedding(name = 'item_pop',
                           input_dim = item_len,
                           output_dim = 1)(item)

# Reshape to be a single number (shape will be (None, 1))
item_pop = Reshape(target_shape = (1, ))(item_pop)
```

```
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4432: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:66: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
```

Next we implement the _User Preferences_, $\theta_{ut}^T \alpha_{c}$, as the dot product of the newly created user embedding $\theta_{u}$ and the item embedding $\alpha_c$. We create a shared embedding layer here that will be used for both indiviual items and the embeddings of items in our baskets variable. This is to ensure that the latent representation of an individual item and the latent representation of that same item in a basket remains the same.
In [0]:

```hl-ipython3
# creating the embeddings for user and item
# Embedding the user (shape will be (None, 1, embedding_size))
user_embedding = Embedding(name = 'user_embedding',
                               input_dim = user_len,
                               output_dim = embedding_size)(user)

# shared item embedding layer for items and baskets
# use mask_zero = True, since we had to pad our baskets with zeros
prod_embed_shared = Embedding(name = 'prod_embed_shared_embedding',
                           input_dim = item_len,
                           output_dim = embedding_size,
                           input_length = None,
                           mask_zero =True)

# Embedding the product (shape will be (None, 1, embedding_size))
item_embedding = prod_embed_shared(item)

# Merge the layers with a dot product along the second axis
# (shape will be (None, 1, 1))
user_item = Dot(name = 'user_item', axes = 2)([item_embedding, user_embedding])

# Reshape to be a single number (shape will be (None, 1))
user_item = Reshape(target_shape = (1, ))(user_item)
```

Jumping forward a bit, we consider the term $\rho_{c}^T(\frac{1}{i-1} \displaystyle\sum_{j=1}^{i-1} \alpha_{y_{tj}} )$ from equation 2. We first create a new item-item interaction vector $\rho_{c}$ that captures complementary effects between items. This is multiplied by the term $\displaystyle\sum_{j=1}^{i-1} \alpha_{y_{tj}}$, which is nothing more than the average of the vectors of all the other items in the shopping basket. Note that this $\alpha$ is the same as the $\alpha$ from our previous embedding. This is why we use a shared embedding layer.
In [0]:

```hl-ipython3
# create item to item embedding
item_item_embedding = Embedding(name = 'item_item_embedding',
                           input_dim = item_len,
                           output_dim = embedding_size)(item)

# embed each item in a basket
basket_embedding = prod_embed_shared(basket)

# take the average of all item embeddings in a basket
avg_basket_embedding = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1))(basket_embedding)
avg_basket_embedding = Reshape(target_shape=(1,embedding_size))(avg_basket_embedding)

# take the dot product of the item and the other items in the basket
# (shape will be (None, 1, 1))
item_basket = Dot(name = 'item_basket', axes = 2)([item_item_embedding, avg_basket_embedding])

# Reshape to be a single number (shape will be (None, 1))
item_basket = Reshape(target_shape = (1,))(item_basket)
```

We now consider estimating price sensitivities. From the SHOPPER algorithm we have, $\gamma_{ut}^T \beta_{c} log(r_{tc})$. We create new item and user embeddings $\gamma_{ut}$ and $\beta_{c}$. In order to calculate the price elasticity for each user, $\gamma$, for each item, $\beta$, we take the dot product of our two embeddings, resulting in $\gamma_{ut}^T \beta_{c}$. This elasticity then gets multiplied with the price $r_{tc}$ to get the overall price effect on shopper utility.

In this step we varied from the SHOPPER paper as we did not normalize the price by taking the log and also do not take the negative. We found our model to be able to recognize that a higher price is bad for a product to be bought and it will learn to lower the utility of a product if the price gets higher. We also estimated the price effects of each individual item, whereas the authours of SHOPPER normalize prices within categories. We did this to get the most granular price effects possible.
In [0]:

```hl-ipython3
# Embedding the product (shape will be (None, 1, embedding_size))
item_embedding_price = Embedding(name = 'item_embedding_price',
                          input_dim = item_len,
                          output_dim = embedding_size)(item)

# Embedding the user (shape will be (None, 1, embedding_size))
user_embedding_price = Embedding(name = "user_embedding_price",
                                 input_dim = user_len,
                                 output_dim = embedding_size)(user)

# Merge the layers with a dot product along the second axis
# (shape will be (None, 1, 1))
user_item_price = Dot(name = 'user_item_price_dot', axes = 2)([item_embedding_price, user_embedding_price])

# Reshape to be a single number (shape will be (None, 1))
user_item_price = Reshape(target_shape = (1,))(user_item_price)

# multiply price effect by price to get effect on utility
user_item_price = multiply([price, user_item_price], name = 'user_item_price')
```

Last we considered the _Seasonality_ $\delta_{wt}^T \mu_{c}$ which we calculate from a new set of item embeddings $\mu_{c}$ and a set of week embedddings $\delta_{w}$. From these latent vectors we again take the dot product to calculate the _Seasonality_.
In [0]:

```hl-ipython3
# Embedding the week (shape will be (None, 1, embedding_size))
week_embedding = Embedding(name = 'week_embedding',
                               input_dim = week_len,
                               output_dim = embedding_size)(week)

# Embedding the product (shape will be (None, 1, embedding_size))
week_item_embedding = Embedding(name = 'week_item_embedding',
                           input_dim = item_len,
                           output_dim = embedding_size)(item)

# Merge the layers with a dot product along the second axis
# (shape will be (None, 1, 1))
week_item = Dot(name = 'week_item', axes = 2)([week_item_embedding, week_embedding])

# Reshape to be a single number (shape will be (None, 1))
week_item = Reshape(target_shape = (1, ))(week_item)
```

Adding these four effects together leads to the utility of a product $c$ to a costumer $u$ when buying this product for price $r_{tc}$ in week $w$ for a certain trip $t$. This utility will be put into the dense layer and with a sigmoid function we get a probability between 0 and 1 of the customer to buy said product.
In [0]:

```hl-ipython3
# Sum up various dot products to get total utility value
x = keras.layers.add([item_pop, user_item, user_item_price, week_item, item_basket])

# Squash outputs for classification
out = Dense(1, activation = 'sigmoid', name = 'output')(x)

# Define model
model = Model(inputs = [user, item, price, week, basket], outputs = out)

# Compile using specified optimizer and loss
model.compile(optimizer = Adam(lr=0.0002), loss = 'binary_crossentropy',
              metrics = ['accuracy'])
```

In [0]:

```hl-ipython3
print(model.summary())
```

```
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
item (InputLayer)               (None, 1)            0
__________________________________________________________________________________________________
basket (InputLayer)             (None, None)         0
__________________________________________________________________________________________________
prod_embed_shared_embedding (Em multiple             3590400     item[0][0]
                                                                 basket[0][0]
__________________________________________________________________________________________________
user (InputLayer)               (None, 1)            0
__________________________________________________________________________________________________
item_embedding_price (Embedding (None, 1, 100)       3590400     item[0][0]
__________________________________________________________________________________________________
user_embedding_price (Embedding (None, 1, 100)       497600      user[0][0]
__________________________________________________________________________________________________
week (InputLayer)               (None, 1)            0
__________________________________________________________________________________________________
lambda_3 (Lambda)               (None, 100)          0           prod_embed_shared_embedding[1][0]
__________________________________________________________________________________________________
user_embedding (Embedding)      (None, 1, 100)       497600      user[0][0]
__________________________________________________________________________________________________
user_item_price_dot (Dot)       (None, 1, 1)         0           item_embedding_price[0][0]
                                                                 user_embedding_price[0][0]
__________________________________________________________________________________________________
week_item_embedding (Embedding) (None, 1, 100)       3590400     item[0][0]
__________________________________________________________________________________________________
week_embedding (Embedding)      (None, 1, 100)       5400        week[0][0]
__________________________________________________________________________________________________
item_item_embedding (Embedding) (None, 1, 100)       3590400     item[0][0]
__________________________________________________________________________________________________
reshape_15 (Reshape)            (None, 1, 100)       0           lambda_3[0][0]
__________________________________________________________________________________________________
item_pop (Embedding)            (None, 1, 1)         35904       item[0][0]
__________________________________________________________________________________________________
user_item (Dot)                 (None, 1, 1)         0           prod_embed_shared_embedding[0][0]
                                                                 user_embedding[0][0]
__________________________________________________________________________________________________
price (InputLayer)              (None, 1)            0
__________________________________________________________________________________________________
reshape_17 (Reshape)            (None, 1)            0           user_item_price_dot[0][0]
__________________________________________________________________________________________________
week_item (Dot)                 (None, 1, 1)         0           week_item_embedding[0][0]
                                                                 week_embedding[0][0]
__________________________________________________________________________________________________
item_basket (Dot)               (None, 1, 1)         0           item_item_embedding[0][0]
                                                                 reshape_15[0][0]
__________________________________________________________________________________________________
reshape_13 (Reshape)            (None, 1)            0           item_pop[0][0]
__________________________________________________________________________________________________
reshape_14 (Reshape)            (None, 1)            0           user_item[0][0]
__________________________________________________________________________________________________
user_item_price (Multiply)      (None, 1)            0           price[0][0]
                                                                 reshape_17[0][0]
__________________________________________________________________________________________________
reshape_18 (Reshape)            (None, 1)            0           week_item[0][0]
__________________________________________________________________________________________________
reshape_16 (Reshape)            (None, 1)            0           item_basket[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 1)            0           reshape_13[0][0]
                                                                 reshape_14[0][0]
                                                                 user_item_price[0][0]
                                                                 reshape_18[0][0]
                                                                 reshape_16[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 1)            2           add_3[0][0]
==================================================================================================
Total params: 15,398,106
Trainable params: 15,398,106
Non-trainable params: 0
__________________________________________________________________________________________________
None
```

In [0]:

```hl-ipython3
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

Out[0]: In [0]:

```hl-ipython3
story = model.fit({'user': X_train['encoded_user'], 'item': X_train['encoded_prods'], 'price':X_train['price'], 'week': X_train['encoded_week'],
                   'basket': basket_prods_train_pad},
          {'output': Y_train},
          epochs=3, verbose=1, validation_split = 0.1, batch_size = 128)
```

```
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3005: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

Train on 8299710 samples, validate on 922191 samples
Epoch 1/3
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:197: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

8299710/8299710 [==============================] - 510s 61us/step - loss: 0.4747 - acc: 0.7654 - val_loss: 0.4142 - val_acc: 0.8043
Epoch 2/3
8299710/8299710 [==============================] - 496s 60us/step - loss: 0.3113 - acc: 0.8658 - val_loss: 0.4158 - val_acc: 0.8135
Epoch 3/3
8299710/8299710 [==============================] - 490s 59us/step - loss: 0.2258 - acc: 0.9101 - val_loss: 0.4576 - val_acc: 0.8131
```

In [0]:

```hl-ipython3
story.history
```

Out[0]:

```
{'acc': [0.7654360212583874, 0.8658132633551283, 0.9100660143545317],
 'loss': [0.4747343215478976, 0.311318981209224, 0.22580939695331692],
 'val_acc': [0.8043236162585331, 0.8135375426568232, 0.8130777680548156],
 'val_loss': [0.414175460535463, 0.41584345185668947, 0.45761839554877254]}
```

In [0]:

```hl-ipython3
# make predictions
preds = model.predict({'user': X_test['encoded_user'], 'item': X_test['encoded_prods'], 'price':X_test['price'],
                       'week':X_test['encoded_week']})
```

In [0]:

```hl-ipython3
X_test.loc[: , 'bought'] = Y_test
X_test.loc[:,'pred_prob'] = preds

# change probabilities to binary classification
# used cutoff of 0.5, because distribution of 1's and 0's is 50-50
X_test['pred'] = round(X_test['pred_prob'])
```

## Results [¶](#results)

After training we take a look at our results. First, we see how the model performed on our prediction task.
In [0]:

```hl-ipython3
# Python script for confusion matrix creation.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
actual = X_test['bought']
predicted = X_test['pred']
results = confusion_matrix(actual, predicted)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :',accuracy_score(actual, predicted))
print('Report : ')
print(classification_report(actual, predicted))
```

```
Confusion Matrix :
[[920699 226066]
 [204955 953756]]
Accuracy Score : 0.8130446814453934
Report :
              precision    recall  f1-score   support

           0       0.82      0.80      0.81   1146765
           1       0.81      0.82      0.82   1158711

    accuracy                           0.81   2305476
   macro avg       0.81      0.81      0.81   2305476
weighted avg       0.81      0.81      0.81   2305476
```

In [0]:

```hl-ipython3
def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score
```

In [0]:

```hl-ipython3
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# plot ROC
compute_roc(X_test['bought'], X_test['pred'], plot=True)
```

Out[0]:

```
(array([0.        , 0.19713368, 1.        ]),
 array([0.       , 0.8231181, 1.       ]),
 0.8129922133759272)
```

## Benchmark [¶](#benchmark)

For our benchmark, we chose a simple logistic regression.
In [0]:

```hl-ipython3
week = pd.get_dummies(final_df['week'],drop_first=True)
year = pd.get_dummies(final_df['year'],drop_first=True)
# create dummy variables for what could be considered categorical
```

In [0]:

```hl-ipython3
final_df.drop(['week','year'],axis=1,inplace=True)

final_df = pd.concat([final_df,week,year],axis=1)

#replace the week and year columns with the dummy variables
```

In [0]:

```hl-ipython3
from sklearn.model_selection import train_test_split

X = final_df.drop(["bought", "article_text", 'basket_hash', 'category_name', 'subcategory_name', 'other_basket_prods', 'other_basket_prods_encoded'],
                  axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, final_df['bought'],
                                                    test_size=0.2,
                                                    random_state=42)
# using a random train/test split (e.g. Option C) with 80:20 ratio
```

In [0]:

```hl-ipython3
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(class_weight = 'balanced')
logmodel.fit(X_train,y_train)
```

Out[0]:

```
LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
```

In [0]:

```hl-ipython3
predictions = logmodel.predict(X_test)

np.mean(predictions)
# 0.46 = the benchmark is predicting about the same amount of 1s and 0s, which is expected
```

Out[0]:

```
0.46087726681700264
```

In [0]:

```hl-ipython3
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
results = confusion_matrix(y_test, predictions)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :',accuracy_score(y_test, predictions))
print('Report : ')
print(classification_report(y_test, predictions))
```

```
Confusion Matrix :
[[228460 167684]
 [200679 199172]]
Accuracy Score : 0.5372295052104599
Report :
              precision    recall  f1-score   support

           0       0.53      0.58      0.55    396144
           1       0.54      0.50      0.52    399851

    accuracy                           0.54    795995
   macro avg       0.54      0.54      0.54    795995
weighted avg       0.54      0.54      0.54    795995
```

In [0]:

```hl-ipython3
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
compute_roc(y_test, predictions, plot=True)
```

Out[0]:

```
(array([0.        , 0.42329052, 1.        ]),
 array([0.        , 0.49811555, 1.        ]),
 0.5374125137114403)
```

# Conclusion [¶](#conclusion)

Our model is computing the probability that the given user will purchase a given product at a given price in a given week, given all the other products in his basket (or other products that the customer came to buy). So circling back to the initial question of designing the coupons that maximize revenue: when the customer enters the store in a given week, we would compute the maximal price this customer is willing to pay (by running the model over the same product but multiple prices and taking the highest price whose probability is still over a certain threshold), and then print coupons for the products that have highest probabilities of purchase under their maximal price. If our model is right, the customer will use all printed coupons during their store visit.

## Suggestions for improvement [¶](#suggestions-for-improvement)

How could we improve this model? Firstly, to make the process of customer choice more believable, we could consider more than one "non-bought" product to be sampled for our model. The dataset would thus become imbalanced, but that is the case in real life - the customer "sees" substantially more products than those that she ends up purchasing. How many she actually considers is questionable.

* Order of items within a basket
* Thinking ahead
* Brand effects
* Daily effects
* Cross-price elasticities

## Other use cases [¶](#other-use-cases)

# Appendix [¶](#appendix)
