+++
title = " Image Captioning"
date = '2018-02-18'
tags = [ "Deep Learning", "Image Captioning", "Class17/18",]
categories = ["seminar"]
banner = "img/seminar/image_captioning/imageCaptioningKlein.jpg"
author = "Class of Winter Term 2017 / 2018"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = " "
+++


# Outline

* [Introduction](#introduction)
* [Data preparation](#data-preparation)
  - Image Features
  - Image Descriptions
* [Setup Caption Generation Model](#setup-caption-generation-model)
  - Model Structure
  - Training Data Generation
  - Train Model
* [Evaluation](#evaluation)
* [Caption Generation](#caption-generation)
* [Summary](#summary)

*Environment Requirements*

To follow this tutorial, we recommend to setup a development environment with Python 3 or higher and assume you to have the following packages installed:

* Keras (with either TensorFlow or Theano backend)
* NumPy
* HU-flickr8k-helper

The last package was developed by ourselves and provides several helper functions to handle the Flickr8k dataset. That way, we can focus on the interesting parts of developing an image caption generation model. For further information, please take a look at the package [documentation](https://www.google.de). Install the package as following:

```
pip install HU-flickr8k-helper
```
And import it at the top of your project:

```python
import HU-flickr8k-helper as huh
```

This tutorial bases on a similar one by [Jason Brownlee](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/), published on his blog.

# Introduction

Short introduction into image captioning and why it may be useful ...

## Dataset

Short introduction of the dataset. Description of its structure (dev/train/test), captions per image, how it was generated, etc. ...

# Data preparation

In this part you will learn how to prepare the image and text data.

## Image Data

In order to work with the image data in the following we need to generate image features out of each image. To do so, we use the VGG 16 CNN developed by Karen Simonyan and Andrew Zisserman [[1]](#references). A pre-trained net saves us a lot of work and time, because we don't need to develop and train an extra model to obtain image features. Especially training would be very time consuming.

Since the VGG 16 model was original developed for image object recognition we have to make some adjustments. After we get the VGG16 object, which is part of the Keras package we need to get rid of the last layer, which is a softmax layer and performs the classification task.

The following `setup_vgg16()` function calls the VGG16 object from Keras, deletes the last layer and fixes the output. As a result it returns the modified VGG16 model.

By calling `extract_feature()` we can generate an image feature using the modified VGG16 model. To do so we have to pass the image to the model calling the `load_img()` function and make some image pre processing which the VGG16 model requires.  
As a result the model returns a 4,096-element vector.

<script src="https://gist.github.com/sim-o-n/0d22dfec7ae2f2d6d7705e94f3035649.js"></script>

The `get_features()` function combines both previous defined methods and offers a handy way to generate features either for multiple images, by passing a directory or for a single image. However the function returns a dictionary which maps the image identifier and its feature.

<script src="https://gist.github.com/sim-o-n/335b352809d8cf75aa7431d6db34bea9.js"></script>

Since it takes some time to generate all features for the dataset, it makes sense to save the dictionary for later use. You can use the `save_features()` function from our helper package to do so. Just pass the dictionary and a path and the features will be saved as a `.pkl` file.

## Text Data
As already describe, the dataset provides five captions for each image and is split in a trainig, test and development set. Each set stors a bunch of image identifier. Let's start by importing the training set. You can easily use the `load_imageID_list()` form the helper package do to so.  

In the next step, load the descriptions form *Flickr8k.lemma.token.txt*. Again, use the `load_descriptions()` function from the helper package and store the descriptions in `raw_desc`. The function takes as parameter:

1. the path to the Flickr8k.lemma.token.txt file
2. the `dataset` object

As the variable name already indicates, we need to pre process the text data before we go on. In order to do so, call the `prepare_descriptions()` function and pass the `raw_desc` as parameter. The function applies standard natural language operations to the data like:

* lower case all words
* remove punctuation like .,-<>()
* remove hanging 's' and 'a'
* remove numbers

This will reduce the size of our vocabulary, which benefits the model performance later. Of cause this also has a downside, since we discard data but we will come back to this in the last part of the tutorial. Save the cleaned data to `cleaned_desc`.

If we would talk about movies we had to add a big **spoiler mark** now, because we have to add something which may no make much sense so fare but will be very important later. Why, we will explain in the part "Training Data Generation". Long story short, we need to wrap each caption into an artificial start and end sequence.  
Let's define a function `wrap_descriptions()`, hand over the `cleaned_desc` and optional a `start` and `end` sequence. If you will not go with the default ones, be aware that these sequences must not already exist in the captions and do not contain spaces.
The function iterates over the descriptions by the image identifier and wraps each description into the start and end sequence and returns the updated dictionary.

Again in order to save some time, let's save the pre-processed and wrapped descriptions. Use the function `save_descriptions()` and hand over the `wrapped_desc` object and a path including a file name.

<script src="https://gist.github.com/sim-o-n/239b99229087a53242af295b81498496.js"></script>

Let's take a quick look into the data. For example choose the first key in the dictionary and inspect how the descriptions changed in the pre-processing process:

<script src="https://gist.github.com/sim-o-n/5c5e2c2dee3afe39647ec93ac151f65c.js"></script>


> ### Take away of this part:
> In this part you learned how to ...
>
> * ... import and use a pre-trained VGG16 model
> * ... generate image features
> * ... load and pre-process image descriptions of the Flickr8k dataset


# Setup Caption Generation Model

In this part of the tutorial you will learn how to develop and implement a deep learning model to generate image captions. We will also take a look at how to modify the text data in order to train the model and finally train it.

But first let's think about how the caption generation process should work in the end. As our goal is not to map an image to a specific caption but rather learn the relationship between image features and word sequences and between word sequences and single words our target is not a complete caption in the dataset but a single word. In other words, the model generates a new caption word by word based on a given image feature and a caption prefix.  
In the beginning each caption only contains the artificial start sequence (which we introduced in the [previous part](#text-data)). We also need the image feature of the image the model should describe in the end. Both, the caption and the image feature vector will be passed to the model. The model will predict the word from the vocabulary which has the highest probability to follow the given caption prefix in combination with the image feature. The predicted word will be appended to the caption and will pass to the model again.  
As you noticed this is an iterative process, which will terminate in two ways. The first stopping criteria would be, that the model predicts the artificial end sequence as the next word. The second one is given by an upper bound of the caption length. We have to specify this bound in advance, which will also have an influence on the models structure later. Either ways, the model will terminate and output the generated caption. We illustrated the process in the flow-chart below.

<center>
  <img src="/blog/img/seminar/image_captioning/Flowchart.jpg" alt="generation-process">
  <p>Figure 1: Flow-chart of the caption generation process.</p>
</center>


## Model Structure

*"What is the most suitable model structure?"*, this may be one of the most important questions when ever developing a neural net. This question is not an easy one and can not easily be answered. A model will always be developed to for a specific task, therefore no general answer can be given.  
However there is some research about how to come up with a good model structure for a given task. In the field of image captioning Marc Tanti, et al. [[2]](#references) compared 16 different model architectures and determined the best one. In this tutorial we will follow their merge-add architecture, which is depicted in Figure XX below:

<center>
  <img src="/blog/img/seminar/image_captioning/theoretic_model_structure.png" alt="theoretic_model_structure" width="60%">
  <p>Figure 2: Merge-add architecture by Marc Tanti, et al..</p>
</center>

This model architecture takes tow vectors as inputs:

1. image feature vector
2. word sequence vector


As you can see, the vectors get injected into two different parts in the model. Marc Tantie, et al. came to the conclusion, that models, where the text data and the image information are handled exclusively perform better. Later in the model both vectors get merged. Since models with a LSTM layer performed slightly better than models with RNN layer in their experiments we will go with this approach.

After the LSTM layer processed the word sequence (in the following caption prefix) its output will be merged with the image feature vector. This merge step can be realized in the ways:

1. *merge-concat*: Concatenate the image feature vector and the caption prefix to one vector. The resulting vector has the length of the sum of length of the input vectors.
2. *merge-add*: Add elementwise image feature vector and caption prefix vector together. The resulting vector has the same dimensions as the input vectors.
3. *merge-mult*: Multiply elementwise the image feature vector and caption prefix together. The resulting vector has the same dimensions as the input vectors.

A clear downside of the concatenate approach is the increase of dimensions of the resulting vector, which results in more complex layer and requires more computational power in the end. In the following we will go with the *merge-add* approach.

The last layer of the model, a softmax layer, finally outputs a probability distribution for all words in the vocabulary. In order to get the next word in the caption prefix we only need to choose the word with the highest probability.

<hr>

Now you learned a bit more about the theoretical background, let's see how we can implement our knowledge in python, using Keras:

Implement the `define_model_structure()` function as following:

<p>
  <strong style="color:#FABC00;">Image feature input</strong>
</p>

The first input takes the 4,096-element image feature vectors (which we generated [here](#image-data)). Therefore its input shape has to be of the same size as the vector. We add a *dropout* layer to prevent overfitting with a dropout rate of 0.5 and compress the vector size to 256 elements in a *dense* layer (using a Rectified Linear Unit (ReLU) function).

<p id="caption-prefix-input">
  <strong style="color:#2980B9;">Caption prefix input</strong>
</p>

The second input takes the caption prefix vector. The shape of this input may vary. Basically it acts as an upper bound of the caption length the model will be able to generate later (see again [here](#setup-caption-generation-model)). Intuitively it would not make sense to train the model to predict longer captions as in the dataset, which represents the ground truth. Use the `get_max_length()` function to get  number of words of the longest caption in the dataset later. The `flat_descriptions()` function is part of our helper package.
The next layer is an *embedding* layer. We need this to handle a zero padding we may need to add to the caption prefix later. This way we can tell the model to ignore the padding, because it does not contain any information for the model to learn. We also increase the dimension of the input vector to 256 elements and apply a *dropout* rate of 0.5 to the data. Finally we pass the data into the *LSTM* layer of the model.

<p>
  <strong style="color:#2ECC37;">Merge and prediction step</strong>
</p>

After the caption prefix vector got processed by the LSTM layer and the image feature vector also got compressed to the same dimensions they get merged by a simple *add* layer. The layer adds both vectors elementwise into a single vector by preserving the original dimensions of the input vectors. Afterwards, the vector gets past into another *dense* layer with a ReLU activation function. The last layer in the model is also a *dense* layer. Instead of a ReLU function it uses a softmax function in order to create a probability distribution over the complete vocabulary. In other words, it predicts the word which has the highest probability to follow the given caption prefix in combination with the image feature. Therefore the size of the output layer has to be equal to the number of unique words in the vocabulary. Use the `get_vocab_size()` function to obtain this number.

As suggested in literature, we use a categorical cross-entropy cost function, since our target variable is in categorical format (see [next part](#training-data-generation) for details) as well as Adam as optimizer method.

<script src="https://gist.github.com/sim-o-n/1b564b79be04302bedea0b84b9bd4381.js"></script>

The handy `plot_model()` function, which comes with Keras, plots any model structure and saves as a picture. This way you can quickly get an overview about a model. This is how our model looks like (we added the boxes):

<center>
  <img src="/blog/img/seminar/image_captioning/keras_model.jpg" alt="theoretic_model_structure" width="60%">
  <p>Figure 3: The model structure.</p>
</center>

> ### Take away of this part:
> In this part you learned how ...
>
> * ... the caption generation process works
> * ... to setup a suitable model with Keras

## Training Data Generation

In the following, we will explain how to generate a training set from the given dataset for our model. In general, neural networks work that way, that they learn to map some data *X* to some target *Y*, e.g. a label (in case Y is known in the training data, this is known as supervised learning).  
In our case *X* consists out of two parts, *X1* the image data and *X2* the captions aka caption prefixes. As our goal is not to map an image to a specific caption but rather learn the relationship between image features and caption prefixes and between caption prefixes and single words our target *Y* is not a complete caption in the dataset but a single word. In other words, the model generates a new caption word by word based on a given image feature and a caption prefix. Therefore the caption generation process is an iterative one. After a new word got predicted by the model it will be appended to the existing prefix and will feed into the model again as already explained in the [Setup Caption Generation Model](#setup-caption-generation-model) part.

*What are X1, X2 and Y?*  

* *X1*: image feature vector
* *X2*: caption prefix
* *Y*: next word

Now let's take the following, pre-processed caption from the dataset. You can think of a table, where each row symbolizes one iteration in the model:

<center>

<p>
  <p style="font-size: 1.5em">"startword person climb up snowy mountain endword"</p>
</p>

| *X1* (image feature) | *X2* (caption prefix)                    | *Y* (next word) |
|----------------------|------------------------------------------|-----------------|
| vec(4,096)           | 0 0 0 0 0 startword                      | person          |
| vec(4,096)           | 0 0 0 0 startword person                 | climb           |
| vec(4,096)           | 0 0 0 startword person climb             | up              |
| vec(4,096)           | 0 0 startword person climb up            | snowy           |
| vec(4,096)           | 0 startword person climb up snowy        | mountain        |
| vec(4,096)           | startword person climb up snowy mountain | endword         |

</center>

You will noticed, that we add zeros to some of the caption prefixes. This is the padding we talked about when [setting up the caption prefix input](#caption-prefix-input) for the model. This is necessary, since each layer of a neural network has a fixed number of input nodes. Therefore any input sequence has to be of the same length.

In the next step we have to transform the caption prefixes and our target *Y* into a machine readable format. For the caption prefixes an easy way is to encode each unique word to an integer and replace the actual words by their numbers. To encode the words we will use a `Tokenizer` object from Keras in the following.  
To encode our target *Y* we will use one-hot encoding. This way we kind of simulate a probability distribution for each word in the vocabulary at each step in the learning process where the probability for one word will be 1 and for all other words 0. After encoding our table may look like this:

<center>

| *X1* (image feature) | *X2* (caption prefix)  | *Y* (next word) |
|----------------------|------------------------|-----------------|
| vec(4,096)           | [0 0 0 0 0 1]          | [0 1 0 0 0 0 0] |
| vec(4,096)           | [0 0 0 0 1 2]          | [0 0 1 0 0 0 0] |
| vec(4,096)           | [0 0 0 1 2 3]          | [0 0 0 1 0 0 0] |
| vec(4,096)           | [0 0 1 2 3 4]          | [0 0 0 0 1 0 0] |
| vec(4,096)           | [0 1 2 3 4 5]          | [0 0 0 0 0 1 0] |
| vec(4,096)           | [1 2 3 4 5 6]          | [0 0 0 0 0 0 1] |

</center>

<hr>

Let's see how we can code this in Python:

To fit a tokenizer on our data implement the `fit_tokenizer()` function, which takes the pre-processed descriptions as input parameter:

<script src="https://gist.github.com/sim-o-n/1c78928c0910bf1d38565e5e4e198935.js"></script>

To automate the previous described procedure we will implement the `create_sequences()` function. Its input parameters are: *a tokenizer object, the preprocessed descriptions, the image features and the length of the longest caption*. Each column will now represented as a `list()`. The first loop iterates over the image identifier. In the second loop each caption of the selected image will first by encode by the tokenizer and than be processed in the third loop as following:

1. split the encoded caption into *X2* and *Y*
2. add zero padding to *X2* (here the `length` parameter comes into play)
3. one-hot encode *Y*
4. append each `list()` with *X1*, *X2* and *Y*

<script src="https://gist.github.com/sim-o-n/a9cd759455c1ae31647a23d81161511d.js"></script>

Let's call the function and inspect its outcome:

<script src="https://gist.github.com/sim-o-n/0b8dc592fa9272e6dc17cd78a2a78d2b.js"></script>

### Train Model

So far we only load training data, to be able to train the model we will also need some test data for cross-validation. You can import and pre-process them the same way as the training data before.

<script src="https://gist.github.com/sim-o-n/ac6a5b2efb5a4dbcd2a67531a8da9892.js"></script>

After we generated the training and validation data we can now train the model.
Call the `model.fit()` function and hand over the *training data, number of epochs, callback and validation data*.  
Keras provides a handy way to monitor the skill of the trained model. At the end of every epoch we will check the skill of the model with the validation set. If the skill improves we save the model. We will do this via the `callback` argument. But first we need to specify where and what Keras should save. Use the `checkpoint` variable for this. We trained our model for 10 epochs, of cause you can choose a higher number. To train for one epoch takes between 30 and 40 minutes, depending on your hardware setting.

<script src="https://gist.github.com/sim-o-n/d4533dca306873bfa0fc845e2c30be93.js"></script>


> ### Take away of this part:
> In this part you learned ...
>
> * ... what a tokenizer does and how to fit it
> * ... how to generate an appropriate data structure to train the model
> * ... how to train the model and monitor it process

# Caption Generation



# Evaluation

# Summary

### References

[1] Karen Simonyan and Andrew Zisserman  
[2] Marc Tantie et al.  
