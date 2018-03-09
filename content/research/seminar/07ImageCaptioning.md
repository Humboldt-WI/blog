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

* Introduction
* Data preparation
  - Image Features
  - Image Descriptions
* Setup Caption Generation Model
  - Model Structure
  - Training Data Generation
  - Train Model
* Evaluation
* Summary

*Environment Requirements*

To follow this tutorial, we recommend to setup a development environment with Python 3 or higher and assume you to have the following packages installed:

* keras (with either TensorFlow or Theano backend)
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

# Introduction

Short introduction into image captioning and why it may be useful ...

## Dataset

Short introduction of the dataset. Description of its structure (dev/train/test), captions per image, how it was generated, etc. ...

# Data preparation

In this part you will learn how to prepare the image and text data.

## Image Data

In order to work with the image data in the following we need to generate image features out of each image. To do so, we use the VGG 16 CNN developed by XXXX. A pre-trained net saves us a lot of work and time, because we don't need to develop and train an extra model to obtain image features. Especially training would be very time consuming.

Since the VGG 16 model was original developed for image object recognition we have to make some adjustments. After we get the VGG16 object, which is part of the keras package we need to get rid of the last layer, which is a softmax layer and performs the classification task.

The following `setup_vgg16()` function calls the VGG16 object from keras, deletes the last layer and fixes the output. As a result it returns the modified VGG16 model.

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
In the beginning each caption only contains the artificial start sequence (which we introduced in the previous part). We also need the image feature of the image the model should describe in the end. Both, the caption and the image feature vector will be passed to the model. The model will predict the word from the vocabulary which has the highest probability to follow the given caption prefix in combination with the image feature. The predicted word will be appended to the caption and will pass to the model again.  
As you noticed this is an iterative process, which will terminate in two ways. The first stopping criteria would be, that the model predicts the artificial end sequence as the next word. The second one is given by an upper bound of the caption length. We have to specify this bound in advance, which will also have an influence on the models structure later. Either ways the model will terminate and output the generated prediction. We illustrated the process in the flow-chart below.

<div>
  <center>
    <img src="https://i.imgur.com/26sU0zm.jpg" alt="generation-process">
    <p>Figure XX: Flow-chart of the caption generation process.</p>
  </center>
</div>

## Model Structure

## Training Data Generation

### Train Model

# Evaluation

# Summary
