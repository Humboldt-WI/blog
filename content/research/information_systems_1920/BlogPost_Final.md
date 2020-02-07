+++
title = "Text Generation TEST"
date = '2019-02-07'
tags = [ "Text Generation", "NLP", "GPT-2", "GloVe", "Text Classification"]
categories = ["Course Projects"]
banner = "img/seminar/sample/hu-logo.jpg"
author = "Lukas FaulbrÃ¼ck, Asmir Muminovic, Tim Peschenz"
disqusShortname = "https-wisample-github-io-blog"
description = "Blog Post for Seminar Information Systems"
+++

# Generating Comments to balance Data

<script src="https://gist.github.com/tim-p95/a3078d7153532821024f6ac13f64bb0b.js"></script>

## Table of Contents (muss angepasst werden)
1. [Introduction](#introduction)
2. [Data Exploration](#exploration)
3. [Data Pre-Processing](#prepros1) 
4. [Text Generation](#textgen1)
    1. [Language Model - GloVe](#textgen2)
      1. [Further Text Preparation](#gloveprep)
      2. [Modeling](#glovemodel)
      3. [Text Generation](#glovegen)
    2. [Language Model - GPT-2](#textgen3)
5. [Byte Pair Encoding](#bpe)
    1. [BPE Introduction](#bpe_intro)
    2. [Byte Pair Encoding in NLP](#bpe_nlp)

6. [Comment Classification Task](#class)
    1. [Relation to Business Case](#class_business)
    2. [Classification Approach](#class_approach)
    3. [Additional Data Preparation for Classification Task](#class_preparation)
    4. [Classification Architecture](#class_architecture)
        1. [RNN Classifier](#class_rnn)
        2. [BOW - Logistic Regression-Classifier](#class_bow)
    5. [Classification Settings](#class_settings)
        1. [(1) Imbalanced](#class_imbalanced)
        2. [(2) Undersampling](#class_undersampling)
        3. [(3) Oversampling GloVe](#class_glove)
        4. [(4) Oversampling GPT-2](#class_gpt2)
    6. [Drawbacks of Oversampling with Generated Text](#class_drawbacks)
    7. [Classification Task - Limitations](#class_limitations)
    8. [Evaluation](#class_evaluation)
    9. [Including the "Not-Sure"-Category](#class_not_sure)
    10. [Different Balancing Ratios for Generated Comments](#class_ratios)
    11. [Conclusion and Discussion of Results](#class_conclusion)
5. [References](#ref1)    



## Introduction <a name="introduction"></a>

In the age of social networks, a strong and active community is an important part for a company to spread its brand. 
Many companies have therefore established a commentary function which serves as a discussion platform. 
Especially newspaper publishers use the commentary function for their online articles. 
A big problem that these companies have to face are comments that are not compatible with their guidelines. 
It includes racist, insulting or hate-spreading comments. 
Filtering all these toxic comments by hand requires enormous human resources. 
One person can review an average of 200 to 250 comments per day. 
With 22,000 comments per day, which is a realistic number, a company needs to pay about 100 employees daily. 
To reduce this number and the associated costs, there is the possibility to build a classifier which predicts whether a comment is approved, whether it should be blocked or whether a human should check it again. 
There are already some scientific papers that have dealt with this very subject (Georgakopoulos et al., 2018; Ibrahim et al., 2018; van Aken et al., 2018). 
With the help of this existing work, companies can build a classifier that is adapted to their needs. 

![alt text](https://drive.google.com/uc?id=1LZGG5UfkWiD4pchZt_q5BKZUH_UAsYbP)

At an average cost of 16€/h per employee and an 8 hour working day, a company must pay a total of 12,800€ per day for 100 employees. If it is possible to reduce the comments to be reviewed to 20%, the company can save a total of 10,240€ in costs per day. This saves on personnel costs, company compliance goals are achieved and the user is also satisfied because her/his comment can be posted directly online.

In the course of the blog a classifier will be built, which should solve the described problem. Only the text is used as input for the model in order to investigate the effects of the text isolated from other features. However, it should be mentioned that other features such as the length of the text, the number of comments posted by a user or the frequency of punctuation marks can also have a positive influence on a classifier.

## Data Exploration <a name="exploration"></a>

In the further work a data set is used, which contains among other attributes the comments, which were posted in the comment section of a German newspaper. In total, the data set contains four attributes. The first attribute shows the date on which the comment was posted. This is used to filter a period from May 2018 to April 2019. The second attribute is a binary indicator about the customer status (subscriber/not subscriber). In order to limit the focus on one target group, only the subscribers are used for further tasks. The third attribute is the main feature and contains the actual text (comment). The last attribute is the label, which we need for the classifier. It indicates whether a comment is toxic or not.

The next figure shows the distribution of the length of the toxic comments. You can see that most of the comments have a length between 10 and 100. 

![alt text](https://drive.google.com/uc?id=1uyONIXgGIX5Xcmu-CclIr7_VxE3Xt-Nj)

A closer look at the distribution of toxic and non-toxic comments shows that there is an imbalance between these two classes.

![alt text](https://drive.google.com/uc?id=1BNVY-wvpI5GaK2flznbSxwkNlL0NdCvk)

In total, we have 1,62 million comments labeled as non-toxic (published) and about 280,000 comments labeled as toxic (not published). Classification of data with imbalanced class distribution has encountered a significant drawback of the performance attainable by most standard classifier learning algorithms which assume a relatively balanced class distribution and equal misclassification costs (Yanmin Sun et al., 2007). There are different ways to deal with unbalanced data. Basic methods for reducing class imbalance in the training sample can be sorted in 2 groups (Ricardo Barandela et al., 2004): 


1.   Over-sampling
2.   Under-sampling






Undersampling is a popular method in dealing with class-imbalance problems, which uses only a subset of the majority class and thus is very efficient. The main deficiency is that many majority class examples are ignored (Liu et al., 2009). Oversampling on the other hand replicates examples in the minority class. One issue with random naive over-sampling is that it just duplicates already existing data. Therefore, while classification algorithms are exposed to a greater amount of observations from the minority class, they won’t learn more about how to tell original and non-original observations apart. The new data does not contain more information about the characteristics of original transactions than the old data (Liu et al., 2007).

To balance our data set, we have decided to use a form of over-sampling. We generate synthetic text using language models that take the minority class as input and add the generated comments to original observations. By this method we don't have duplicates of the observations from the minority class, but new comments that take the same properties of the minority class (Stamatatos, 2007).

![alt text](https://drive.google.com/uc?id=15c4wJrdmQRa0KWAcbA3mLrLfnt0Hz_WC)

It is not necessarily the best way to balance the data set so that the final ratio is 50:50. Sometimes it can increase the performance of the classifier by balancing the ratio of the classes only minimally.

When generating the text we decided to compare two language models with each other. One is a keras language model, used in generation and training. It is sequence to sequence model using a bilayer encoder and decoder with dropout and a hidden dimension of 128. The second language model we are using is GPT-2. The Generative Pre-Training version 2 was developed by OpenAI and is a state-of-the-art language model that can generate text. A detailed explanation of GPT-2 will be given in a later part of the blog.

## Data Pre-Processing<a name="prepros1"></a>

Before the generation of comments can begin, the data must first be prepared. We will not explain all pre-processing steps, but focus on the central ones.

First we split our data into a train and test set.


```python
train_set, test_set = train_test_split(data,
                                       test_size = 0.2, 
                                       random_state=11)
```

The test set won't be touched until the very end when we are doing the evaluation.

A key to generating good comments is the preparation of the text. For this purpose we have defined two functions. One function prepares the text for the generation part and the other function for the classification.


```python
# Defining a function for text preparation for text generation

# Detokenizer combines tokenized elements
detokenizer = TreebankWordDetokenizer()

def prepare_text_gen(text):
    
    # Remove URLs
    text = re.sub(r'https\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)
    
    # Remove references to Usernames
    text = re.sub(r'@\S+', '', text)

    # Remove line breaks
    text = re.sub("\n|\r", ' ', text)
  
    # Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Devide the text into sequences of words and transform the tokens into lower case. This avoids having multiple copies of the same word.
    text = word_tokenize(text.lower())
    
    # Remove the non alphabetic tokens.
    text = [token for token in text if token.isalpha()]

    # Remove the punctuation since it doesn't give any extra information while treating the text. Furthermore, it helps us to reduce the size of the training data.
    table = str.maketrans('', '', string.punctuation)
    text = [token.translate(table) for token in text]

    return detokenizer.detokenize(text)
```


```python
# Defining a function for text preparation for classification
stop_words = set(stopwords.words('german'))

stop_words = list(stop_words)

# Filter certain stop words from the list e.g. kein, nicht
sw_remove = ['kein', 'keine', 'keinem', 'keinen', 'keiner', 'keines', 'nicht', 'nichts']

for x in sw_remove:
    if x in stop_words:
        stop_words.remove(x)

stop_words = set(stop_words)

def prepare_text_class(text):
    
    # Remove URLs
    text = re.sub(r'https\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)
    
    # Remove references to Usernames
    text = re.sub(r'@\S+', '', text)
    
    # Remove line breaks
    text = re.sub("\n|\r", ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    # devide the text into sequences of words and transform the tokens into lower case. This avoids having multiple copies of the same word.
    text = word_tokenize(text.lower())
    
    # remove the commonly occuring words. For this purpose, we use predefined libraries. In the same step we remove the non alphabetic tokens.
    text = [token for token in text if token not in stop_words and token.isalpha()]

    # remove the punctuation since it doesn't give any extra information while treating the text. Furthermore, it helps us to reduce the size of the training data.
    table = str.maketrans('', '', string.punctuation)
    text = [token.translate(table) for token in text]

    return detokenizer.detokenize(text)
```

The different steps in the preparation of the text can be discussed. Especially the removal of punctuation marks can lead to worse results when generating new comments. For the blog post, we stick to the steps shown. But for future research it can be experimented with different text pre-processing steps.

The following word cloud shows the most used words in the class of toxic comments after cleaning and removing some common words (to get a clearer word cloud).

![alt text](https://drive.google.com/uc?id=1V-edDavqxOa5sFLcvrYhyYH_C6QwBPhO)

While exploring and pre-processing the text, we have noticed some points that can have a negative impact on the performance of our models. Not all comments that are labeled as toxic can be guaranteed to be so. We have read some comments where, without knowing the context, it was difficult to understand why they were labelled as toxic. For example, the comment 'Und wer ist der Papa?' ('And who is the daddy?') may not look toxic at first glance, but may be inappropriate in the context. Another critical point is the vocabulary of the users. Some words are used which only occur in the cosmos of the comment section. This can become a problem if a pre-trained embedding is used which was trained in another context. The last point, which can have a negative impact, are many spelling mistakes made by the users. Pre-trained embeddings will probably not recognize words with spelling mistakes and self-trained embeddings will represent these words as individual vectors. For now we ignore that potential problems but have to keep that in mind for evaluation as well as for future steps.

## Text Generation<a name="textgen1"></a>

For the text generation we compare two language models. In order to better understand where we are in the world of embeddings, we have marked the two important places in the following graphic in red. The graphic was created by Andrien Sieg which blog post you can find here [link](https://towardsdatascience.com/from-pre-trained-word-embeddings-to-pre-trained-language-models-focus-on-bert-343815627598).

![alt text](https://drive.google.com/uc?id=1N-j-F884yF0ABFMgGe6zhPB8xH-hYFai)

### Language Model - GloVe<a name="textgen2"></a>
The first model is a sequence to sequence model. It takes a sequence as an input and generates a new sequence (the words of our new comment). A sequence to sequence model has two components, an encoder and a decoder. The encoder encodes the input sequence to a vector called the context vector. The decoder takes in that context vector as an input and computes the new sequence. A disadvantage of this structure is that the context vector must pass all information about the input sequence to the decoder. So that the decoder is not only dependent on the context vector, but gets acess to all the past states of the encoder, we implement an additional Attention Layer, which takes over this task. The function and idea of the Attention Layer, which we use here, was introduced by Bahdanau et al. (2014). How Attention works in detail is explained in a later part of the post when GPT-2 takes it's turn.

In order to represent our words we use a pre-trained German GloVe embedding from deepset [link](https://deepset.ai/german-word-embeddings). It was trained on a German wikipedia corpus in 2018. The following graphic visualizes the connections of the words in space. The visualization is limited to 50 words.

![alt text](https://drive.google.com/uc?id=194LFyOAILghuaht1ySr4xKPFzYkVWuaB) 

We will not give a detailed description of embeddings in our blog, as this topic is a blog post in itself, but a very good explanation is given by Allen and Hospedales (2019), which can be found in the references.

For our first model some additional steps have to be taken to prepare the input for the model. We set a training length at 15. Furthermore, we turn our texts into sequences of integers. These steps and some more are shown and briefly explained in the following function.


```python
def build_sequences(texts,
                   training_length=15):

    # Vectorizing the text samples into a 2D integer tensor
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    # Create dictionaries to look up the words and the other way around
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts

    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)

    # Limit to sequences with at least 30 words
    sequence_lengths = [len(x) for x in sequences]
    over_idx = [
        i for i, l in enumerate(sequence_lengths) if l > (training_length + 15)
    ]

    abstracts = []
    new_sequences = []

    # Keep the sequences with more than 30 words
    for i in over_idx:
        abstracts.append(texts[i])
        new_sequences.append(sequences[i])

    features = []
    labels = []

    # Iterate through the sequences of tokens
    for seq in new_sequences:

        # Get multiple training samples from each sequence
        for i in range(training_length, len(seq)):
            # Extract the features and label. The label is the 16th word in the training example.
            extract = seq[i - training_length:i + 1]

            # Set the features and labels
            features.append(extract[:-1])
            labels.append(extract[-1])

    # Return all objectes needed for the model
    return word_idx, idx_word, num_words, word_counts, abstracts, new_sequences, features, labels
```

Next we need to create the features and labels for both the training and validation. Since our model will be trained using categorical cross entropy we need to convert the labels in one hot encoded vectors.


```python
def create_train_valid(features,
                       labels,
                       num_words,
                       train_fraction):

    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=50)

    # Decide the ratio between training and validation. We decided for 70:30 ratio.
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    return X_train, X_valid, y_train, y_valid
```

The next step is loading the German GloVe embedding and creating an embedding matrix. In addition, we check how many words without pre-trained embeddings exist. In total, about 40% of the words in the comments are not covered by the embedding. That are a lot of words, which leads to a loss of information. The reason for this are the problems already mentioned at the end of the exploring part. The context in which the embedding was trained (wikipedia) is different from the vocabulary of the users in our comment section. Furthermore, words with spelling mistakes do not exist in the embedding and are counted as separate words, even if the correctly spelled word occurs in the embedding. A solution to this problem is to train an own embedding based on the comments, which can be expensive. However, for the generation of the text we continue with the pre-trained embedding.


```python
embeddings_index = {}
f = open('german_glove.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

not_found = 0
embedding_matrix = np.zeros((num_words, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        not_found += 1

print(f'There were {not_found} words without pre-trained embeddings.')
```

After we have completed all the necessary steps, we can start building the architecture of our recurrent neural network.

After converting the words into embeddings we pass them to our encoder and decoder block. The encoder consists of a bidirectional lstm layer and another lstm layer with 128 nodes each. LSTM (Long Short Term Memory) is a type of recurrent neural network introduced by Hochreiter & Schmidhuber (1997). As an RNN it has a word as an input instead of the entire sample as in the case of a standard neural network. This ability makes it flexible to work with different lengths of sentences. The advantage of LSTM against RNN is the ability to capture long range dependencies. Since we are interested in the whole context of a comment that memory is very useful. Bi-Directional brings the ability to work with all available information from the past and the future of a specific time frame (Schuster and Paliwal, 1997). The decoder consists of two lstm layers also with 128 nodes. Both the encoder and the decoder use dropout to prevent overfitting. Return_sequences is set to true so that the next layer has a three-dimensional sequence as input. Next we give the attention layer as input the output from the encoder and decoder. The output of the attention layer must then be merged with the output of the decoder. Before we can feed the output to our final layer we have to make sure that it has the right shape. Our last layer is the prediction layer using softmax as an activation function.

For the loss function we have chosen categorical crossentropy. It is used in classification problems where only one result can be correct. 

![categorical crossentropy math](https://drive.google.com/uc?id=1YEOSL6FpfGdhcVUR3AFdysZ0j94Hrdvt) 

Where:

M - number of classes

N -  number of samples

ŷ - predicted value

y - true label

log - natural log

i - class

j - sample

Categorical crossentropy compares the distribution of the predictions (the activations in the output layer, one for each class) with the true distribution, where the probability of the true class is set to 1 (our next word) and 0 for the other classes (all the other words). The closer the model’s outputs are to the true class, the lower the loss. 




As an optimizer we use Adam optimizer (Adaptive Moment Estimation) which computes adaptive learning rates for each parameter. That algorithm is used for first-order gradient-based optimization of stochastic objective functions (Diederik P. Kingma and Jimmy Ba, 2014). As an evaluation metric we use accuracy.


```python
x = Input(shape=(15,))

embedding = Embedding(input_dim=num_words,
                      output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      trainable=False)(x)

encoder = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding)
encoder = LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(encoder)

decoder = LSTM(128, recurrent_dropout=0.35, dropout=0.3, return_sequences=True)(embedding)
decoder = LSTM(128, recurrent_dropout=0.35, dropout=0.3, return_sequences=True)(decoder)

attention = Attention()([encoder, decoder])

decoder_concat_input = Concatenate()([decoder, attention])

flatten = Flatten()(decoder_concat_input)

y = Dense(num_words, activation='softmax')(flatten)

model = Model(inputs=x, outputs=y)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
```

Now that we have built our model we can train it for a fixed number of epochs. We use callbacks to save the best model and to stop the training if there is no improvement of the validation loss after 5 epochs. After we have trained our model we evaluate it.


```python
cp = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ea = EarlyStopping(monitor="val_loss", mode="min", patience=5)

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=2048,
    verbose=1,
    callbacks=[cp,ea],
    validation_data=(X_valid, y_valid))

model.evaluate(X_valid, y_valid, batch_size=2048, verbose=1)
```

With our first model we achieve a cross entropy of 7.2, which can be interpreted as average performance. Very good text generation is achieved with a cross entropy of about 1.0.

After we have finished training our neural network, we can start generating text. To do this, we first choose a random sequence, from the sequences we have defined in one of the previous steps. From this sequence we use 15 contiguous words as seed. This seed is then used as input for the model to predict the 16th word. The 16th word is appended to the last 14 words of the seed, so that we again have 15 words as input for the model to predict the 17th word. This procedure is continued until a length is reached that matches the length of a randomly selected toxic comment. For the choice of the generated word, we have also implemented temperature. It defines how conservative or creative the models's guesses are for the next word. Lower values of temperature generates safe guesses but also entail the risk that the same words occur frequently or that these words are repeated. Above 1.0, more risky assumptions are generated, including words that increase the diversity of the generated comment.


```python
def generate_output(model,
                    sequences,
                    training_length=15,
                    temperature=1):

    # Choose a random sequence
    seq = random.choice(sequences)

    # Choose a random starting point
    seed_idx = random.randint(0, len(seq) - training_length - 10)
    # Ending index for seed
    end_idx = seed_idx + training_length

    generated = []

    # Extract the seed sequence
    seed = seq[seed_idx:end_idx]

    # Select length from a random toxic comment
    gen_length = text_df.word_count[int(random.uniform(0, 1)*len(text_df.word_count))-1]        
        
    # Adding new words
    for i in range(gen_length):

        # Make a prediction from the seed
        preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(np.float64)

        # Temperature for sampling function. Higher values increase selection of less-likely characters.
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)

        # Softmax
        preds = exp_preds / sum(exp_preds)

        # Choose the next word
        probas = np.random.multinomial(1, preds, 1)[0]

        next_idx = np.argmax(probas)

        # New word gets added to seed
        seed = seed[1:] + [next_idx]
        generated.append(next_idx)

    # Getting the words of generated text
    gen_text = []

    for i in generated:
        gen_text.append(idx_word.get(i))           
                    
    return gen_text
```

With the help of the defined function, new comments are created and saved as csv in order to use them later for classification.


```python
generated_text = []
for i in range(20000):
    gen_text = generate_output(model, sequences)
    generated_text.append(gen_text)
    if len(generated_text)%1000 == 0:
        print('1k')
generated_text_df = [' '.join(i) for i in generated_text]
generated_text_df = pd.DataFrame(generated_text_df, columns = ['text'])
generated_text_df.to_csv("generated_text_glove.csv", index=False)
```

One question that needs to be asked after text generation is what metric is used to evaluate the quality of the text. One of the most popular metrics for evaluating sequence to sequence tasks is Bleu (Papineni et. al, 2002). The basic idea of Bleu is to evaluate a generated text to a reference text. A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0. However, it has some major drawbacks especially for our use case. A bad score in our case does not necessarily mean that the quality of our generated comment is poor. A good comment looks as if it was written by a human being. To check the quality of the comments, we decided to read some of them randomly. Two samples for a good and a bad one are shown after the GPT-2 part.

# Byte Pair Encoding <a name="bpe"></a>

## BPE Introduction <a name="bpe_intro"></a>

In this section we aim to explain why GPT-2 is able to recognize German words, although trained solely on English text data from the web. To understand why GPT-2 can be used to generate German text, it is necessary to understand the functioning of byte pair encoding (BPE) in NLP. 

Originally, BPE was developed in 1994 by Philip Gage, as a comparatively simple form of data compression. The main idea was to replace the most common pair of consecutive bytes in a data file with another byte, which does not belong to the set of occurring characters. Thus, the BPE-algorithm checks the whole text corpus for multiple occurrences of character-/(byte-)pairs and converts them to another character/byte. Thereby, the size of the original data is compressed by reducing the amount of characters that is necessary to represent that data. In order to decompress/decode the compressed/encoded data again, it is necessary to provide a table of executed replacements. This table allows to rebuild the original data by mapping the replacement characters back to their original representations (character pairs). The original and the decompressed data should be equal at the end, to ensure consistency.

Hence, BPE allows to reduce the length of character sequences by iterating through the text corpus several times. The algorithm can reduce the size of the data as long as a consecutive byte pair occurs at least two times. Since BPE is a recursive approach, it allows further to replace also multiple occurrences of replacement character pairs. This can be used to even encode longer sequences than only two bytes with only one single replacement character. Although the BPE-algorithm is not the most efficient form of data compression, it offers some valuable aspects for the use in NLP.


Bild Data Compression

## Byte Pair Encoding for NLP <a name="bpe_nlp"></a>

The idea of BPE can be repurposed for the use in NLP applications. One main aspect in this regard is the reduction of the vocabulary size. To achive this, the original BPE algorithm needs to be modified for the use on natural language text. The reduction of the vocabulary size is realized by using a subword tokenization procedure, which is inspired by BPE. 

In general, there are three levels, in which words can be included into a vocabulary (see figure below). The first is the word level representation where only complete words are used in the embedding. For each word, a vector is calculated to capture its semantics. This generally leads to large embeddings, depending on the amount of text and heterogeneity of words included. The second option for determining vector representations of words inside an embedding is to reduce words to character level. This significantly reduces the size of the vocabulary since only the set of contained characters is left. However, the usage of character level word representations in NLP resulted in comparatively poor outcomes in some applications as shown e.g. in Bojanowski et al. (2015). The third option is to use subwords instead of character or word level representations. This offers a good balance between both previous types of word representations by combining the advantage of a reduced vocabulary size with a more "meaning-preserving" word splitting.

Bild Word Splitting Options

Additionally to the reduction of vocabulary size, there are several more advantages of splitting words into subwords. These are primarily important for our application, i.e. the generation of German text by using an “English”-language model (GPT-2). First, the language model does not fall into out-of-vocabulary-error, when a German word is presented to it. Words from the German vocabulary can be perceived as unknown to the vocabulary of the English language model. Although GPT-2 was not provided with German text during training, there exists a number of subwords that are equal in German and English. An example is given in the figure below. Since single characters are also included into the GPT-2 vocabulary, German words that cannot be constructed out of multi-character subwords are simply represented at character level. 




Bild BPE Steps

Another useful advantage of splitting words into subword tokens is the fact that rare words can potentially be returned in the output of the generation process. When using word level embeddings, it is generally necessary to reduce the vocabulary size by setting a fixed limit to the number of words. This means that very rare words, that does only occur e.g. one or two times in the whole corpus, drop out of the set of words that can appear in the generated text. By splitting rare words into much more frequent subwords, these can still be included in the resulting synthetic text. 

A major modification that is necessary to use BPE in NLP applications is that pairs of subwords (bytes) are not replaced by another character, but rather are merged together to a new subword. This is done for all instances of a pair in the whole text. The process can be repeated as long as all words are broken down into segments. Therefore, the subwords that are included in the resulting vocabulary are dependent on the underlying text corpus. The GPT-2 vocabulary consists of approximately 50.000 subword tokens. It is possible to set certain parameter values that limit the merging of subword pairs, depending on their frequency of occurrence. The NLP-adapted BPE algorithm does not fulfill the original task of data compression, since it does not reduce the amount of raw data. It rather represents a splitting heuristic that is able to adjust the subword vocabulary to the text it is applied to.

Bild "Unfriendly"

The BPE-based subword tokenization helps us to explain, why German words are part of the GPT-2 vocabulary and does not lead to an out-of-vocabulary error. However, since the meaning of subword tokens are in most cases completely different across both languages, it cannot be explained how pre-trained language information from GPT-2 can be transferred to our German text generation process. It is not even clear whether any semantical or gramatical knowledge is transferred at all. For example, the German word “die” as female article and the English verb “(to) die” have a completely different meaning. Therefore, the pre-trained word vectors from GPT-2 are expected to be more or less useless to capture the meaning of German subwords from our comment texts (although exceptions exist, e.g. alphabet, film, hotel). That means that the majority of knowledge about grammatical and semantic formations of the German language must be learned during the fine-tuning of the GPT-2 model.

Bild BPE englisch deutsch

# Comment Classification Task <a name="class"></a>

## Relation to Business Case <a name="class_business"></a>

In our business application scenario above, we described that a large number of employees is necessary if we want to check each incoming comment manually. Due to legal regulations and business compliance goals, there is no opportunity to not check every individual comment.

Our aim is to tackle this issue by training a classification model on existing comments that were already manually classified as publishable or not. The model should be capable to differentiate between comments that violate any of the given restrictions, and those that do not. This leads us to a binary classification setting, in which we want to predict a probability of being publishable for each comment. 


## Classification Approach <a name="class_approach"></a>

In our classification stage, we train several models on different data inputs. With the resulting probability predictions, we can decide how to deal with the comment and evaluate the model performance. Since our data is imbalanced with a ratio of approx. 85:15 (majority class: published comments), we face different classification settings that are explained below. We further used two different classifiers in our application to allow for a comparison of the results among different classification approaches. The architecture of the classifiers is described later on.

## Additional Data Preparation for Classification Task <a name="class_preparation"></a>


The preparation of our comment text data for the classification included all the steps from the cleaning phase of the text generation. Since the requirements in terms of text cleaning are different for both approaches in which we use our comments (i.e. generation and classification), we have to make some additional modifications to the text cleaning to predict publishing probabilities. 

First, we replaced German Umlaute (ä, ö, ü) with equivalent expressions (ae, oe,ue) to prevent errors in the following stages of the classification task.



```python
import re

#replace ä -> ae, ö -> oe, ü -> ue 
def replace_umlaute(text_df_col):
    text_df_col = text_df_col.apply(lambda x: re.sub(r'ä', 'ae', str(x)))
    text_df_col = text_df_col.apply(lambda x: re.sub(r'ö', 'oe', str(x)))
    text_df_col = text_df_col.apply(lambda x: re.sub(r'ü', 'ue', str(x)))

    return text_df_col

text_classification_train_reduced['text'] = replace_umlaute(text_classification_train_reduced['text'])
text_classification_test_reduced['text'] = replace_umlaute(text_classification_test_reduced['text'])
```

Next, we applied stemming to our words from the text corpus using the German “Snowball”-Stemmer, provided by nltk. The target was to align words with different grammatical representations having the same meaning. Stemming was applied to the test and training data. A coding example for the training data is given below.


```python
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("german")

#stemming for reduced training data
nrows = len(text_classification_train_reduced)
stem_text_list = []

for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    stem_list = []
    
    # Save the text and its words into an object
    text = text_classification_train_reduced.iloc[row]['text']
        
    # Join the list
    stem_text = stemmer.stem(text)
    
    # Append to the list containing the texts
    stem_text_list.append(stem_text)

text_classification_train_reduced['text'] = stem_text_list
```

To further reduce the number of words, we removed stop words from the text corpus for classification. We modified the available list of German stop words to our needs. Some words that are included in our pre-defined stop word list like “nicht” (not) or “kein” (no, not a …) were excluded from that list. The occurrence of these negation words in a comment might have an impact on the meaning of a sentence.

## Classification Architecture <a name="class_architecture"></a>

### RNN Classifier <a name="class_rnn"></a>

For our binary classification problem, we first construct a recurrent neural network (RNN), of which the model architecture is depicted below. We included an embedding layer into our model, which uses a Word2Vec embedding that was trained on 1 million comments from our own text corpus. We further included a bidirectional LSTM Layer in our model to capture the sequential character of our data. To obtain the required output from our RNN model, we included two dense layers. The first uses relu and the second sugmoid as activation function.

Image Model Summary

### BOW - Logistic Regression-Classifier <a name="class_bow"></a>

Additionally to our RNN-classifier, we trained a comparatively simple Bag-of-Words (BOW) Classification model, which uses logistic regression to predict the probability for a comment to be publishable. This second model is also trained with the same data composition/distribution as the RNN-model, in each of the four classification scenarios. The second classifier serves as a benchmark model to evaluate the performance of our RNN-model, in each of the classification settings. Although a comparatively simple approach, the combination of BOW and logistic regression has shown to yield at good results in similar applications (Sriram et al., 2010). 

## Classification Settings <a name="class_settings"></a>

The classification settings vary in terms of the data we input to our two models. Since our main goal is to examine the use of generated comments to balance textual data, we need a benchmark to measure the impact of our synthetic comments. In total we ended up with four different classification settings, that can be divided into either benchmark (imbalanced, undersampling) or target (both settings including generated comment data). 

We used both our classifiers in each of the settings that are described in the following part. For the first setting (Imbalanced) an extended explanation of the code is given below to describe our classification approach more in depth. Since the classification architecture is equal for all settings, we do not describe the code for the reamining ones. However, we will give code examples for the data preparation that varies for the other settings.

### (1) Imbalanced <a name="class_imbalanced"></a>

In the imbalanced setting, we use the cleaned comment text data to train our models. Hence, the classifiers are provided with the imbalanced comment data from the original data set. We did not change the distribution of publishable and non-publishable comments. The imbalanced setting is meant to serve as a first benchmark for our following classification settings. 

The implementation of the Bag of Words (BOW) Classifier is constructed as shown in the code below. We first count word occurrences using the CountVectorizer from the scikit-learn package. Next, we run a logistic regression to estimate the relations between word occurrences and the publishing status of our comments. The 'bow_classification'-function returns the predicted probabilities of being published for the comments in the test data. 


```python
def bow_classification(comment_train, comment_test):

    from sklearn.feature_extraction.text import CountVectorizer

    #count word occurrences
    vectorizer = CountVectorizer()
    vectorizer.fit(comment_train)

    #prepare training and test data for BOW model
    X_train_bow = vectorizer.transform(comment_train)
    X_test_bow  = vectorizer.transform(comment_test)

    ### logisitc regression:
    from sklearn.linear_model import LogisticRegression

    #train logisitc regression model
    classifier = LogisticRegression()
    classifier.fit(X_train_bow, label_train)

    #predict class probabilities
    pred_bow = classifier.predict_proba(X_test_bow)
    pred_bow = pred_bow[:, 1]

    return pred_bow
```

For each model in all of our settings we calculate evaluation metrics based on our model predictions. We convert our probabilities to class predictions, using a threshold of 0.5. Next, we build a confusion matrix for each model using the pandas crosstab function. These confusion matrices are used during the evaluation phase to compare the models accross settings.


```python
def conf_matrix(pred_class_df_col, true_class_df_col):
    #prepare data
    cm_data = {'class_pred': list(pred_class_df_col),
               'class_true': list(true_class_df_col.astype(int))}
    
    #create dataframe
    cm_df = pd.DataFrame(data = cm_data, columns=['class_pred', 'class_true'])

    #create confusion matrix
    cm = pd.crosstab(cm_df['class_true'], cm_df['class_pred'], rownames=['True'], colnames=['Predicted'])

    return cm
```

Furthermore, we are able to calculate additional metrics that allow us to evaluate the performance of our binary classification models. We use sklearn.metrics functions to calculate F1-Score, the area under receiver operating characteristics (ROC) curve (AUC) and the accuracy of our predictions.


```python
import sklearn.metrics

f1_bow_imbalanced = sklearn.metrics.f1_score(text_classification_test_reduced.published, pred_bow_imbalanced_class.class_pred)
auc_bow_imbalanced = sklearn.metrics.roc_auc_score(text_classification_test_reduced.published, pred_bow_imbalanced_class.prob_pred)
acc_bow_imbalanced = sklearn.metrics.accuracy_score(text_classification_test_reduced.published, pred_bow_imbalanced_class.class_pred)b
```

For the recurrent neural network classifier (RNN), some additional preparation steps are neccessary to fit the model. First, we perform tokenization using the text tokenizer from keras. Second, we are padding the sequences to a fixed length, which we set to 100.


```python
def prepare_data_rnn(comment_train, comment_test):
    
    ### Tokenization:
    from keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(comment_train)

    X_train = tokenizer.texts_to_sequences(comment_train)
    X_test = tokenizer.texts_to_sequences(comment_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    ### Padding Sequences
    from keras.preprocessing.sequence import pad_sequences

    #setting maximum length of sequence
    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    return X_train, X_test
```

After preparing the data, we configured our RNN classifier. Therefore, we first set our parameter values. Since they have to be equal for each setting, it is practical to set them globally at a fixed position in the code.


```python
#setting RNN parameters for all models
LSTM_NODES = 64
DROPOUT = 0.1
RECURRENT_DROPOUT = 0.1
DENSE1_NODES = 64
DENSE2_NODES = 1
EPOCHS = 20
BATCH_SIZE = 512
```

Then we configured our model architecture, which was already described in the previos section. We used keras for our RNN implementation. As optimizer for the model we took the adam optimizer and since our target is a binary classification, we used binary crossentropy for loss minimization.


```python
#using RNN model with Embedding and bidirectional LSTM layer for classification
from keras.models import Sequential
from keras import layers
from keras.layers import LSTM, Dropout
from keras.layers import Bidirectional
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping

#setting embedding dimensionality
embedding_dim = 300

#initialize RNN model
model = Sequential()

#add embedding layer
model.add(layers.Embedding(input_dim=num_words, 
                           output_dim=embedding_dim, 
                           embeddings_initializer = Constant(embedding_matrix_word2vec),  
                           trainable = False))

#add bidirectional LSTM layer
model.add(Bidirectional(LSTM(LSTM_NODES, dropout = DROPOUT, recurrent_dropout = RECURRENT_DROPOUT)))

#add additional dropout layer
model.add(Dropout(DROPOUT))

#adding dense layers to produce required output probability
model.add(layers.Dense(DENSE1_NODES, activation='relu'))
model.add(layers.Dense(DENSE2_NODES, activation='sigmoid'))


#complite the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
```


```python
#train RNN model
ea = EarlyStopping(monitor="val_loss", mode="min", patience=5)

model.fit(X_train, label_train,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(X_test, label_test),
          callbacks=[ea],
          batch_size=BATCH_SIZE)
```

After training the RNN model, we predict class probabilities for our test data. Similar to the BOW model case, we calculated the confusion matrix and our evaluation metrics for the RNN model.

### (2) Undersampling <a name="class_undersampling"></a>

The second benchmark setting of our classification part consists of a simple undersampling approach. We reduce the number of comments from our majority class (published comments) to balance our data set. Since our data is imbalanced with a ratio of 85:15, we need to exclude a significant amount of published comments, in order to align them to our minority class (not published comments). Obviously, the disadvantage in this setting is that we hide available information from our classifiers that might include useful information to identify publishable comments. We use this setting as a second benchmark because, in contrast to our imbalanced setting, we now have a balanced situation that we want to achieve also in our target settings by including synthetic comments. 

Both our benchmark settings are useful to evaluate the impact of our generated comments on the outcome of the classification. The imbalanced case allows us to compare the outcomes between an imbalanced and a balanced text classification task. The amount of original data is not changed, but the distribution of both classes is different. The undersampling scenario can be used to compare the outcomes of two classification tasks using a balanced data set. Here, we change the amount of original comments by reducing the number of observations from the majority class, but the distribution of both classes is balanced. 



```python
#perform undersampling

#extract number of not published comments
count_not_pub_comments = len(text_classification_train_reduced[text_classification_train_reduced['published'] == False])

#get indices of published comments
idx_pub_comments = text_classification_train_reduced[text_classification_train_reduced.published == True].index

#random sample published comments indices
random_indices = np.random.choice(idx_pub_comments, count_not_pub_comments, replace = False)

#get indices of not published comments
idx_not_pub_comments = text_classification_train_reduced[text_classification_train_reduced.published == False].index

#concat non-published indices with sample published ones
under_sample_indices = np.concatenate([idx_not_pub_comments, random_indices])

#construct balanced dataframe
df_under_sample = text_classification_train_reduced.loc[under_sample_indices]
```

### (3) Oversampling GloVe <a name="class_glove"></a>

In both our oversampling scenarios (GloVe and GPT-2), we use synthetic comments of our underrepresented class (not published comments) to balance our comment type distribution. The advantage of this approach is that we can use all of the available original text data for the classification. In contrast to the undersampling case, we do not need to exclude any comments, but still reach a situation where we have a balanced distribution of published and unpublished comments. Hence, we provide our classifiers with the original data and the synthetic comments that were generated by using the pre-trained German GloVe embedding. 

The generated text data is also edited using the cleaning function for classification from above. Then we extend the original data with the synthetic comments of the underrepresented class (not published) and the according label for the publishing status.

### (4) Oversampling GPT-2 <a name="class_gpt2"></a>

Our second oversampling scenario is conceptionally equal to the first one using GloVe embedding during the text generation step. The only change in this scenario is the use of the pre-trained GPT-2 language model for the creation of synthetic comments instead of the GloVe-based generation architecture. A major difference between GPT-2 and GloVe in our application is that they were trained on text data of different languages. GPT-2 was trained on massive amounts of English text data. GPT-2 has advanced capabilities in the generation of English text, by fine-tuning it to domain specific textual data. We want to make use of this generative power of GPT-2, but since our data set consist solely of German text, we face a severe drawback regarding the language-related barrier. On the other hand, the GloVe embedding we use was trained on German text only. Therefore, we do not need to overcome the language differences. However, text generation using a GloVe embedding might not have the same generative power as GPT-2. Thus, we have a trade-off between language-related differences and the generative capabilities of both generation approaches.

## Drawbacks of Oversampling with Generated Text <a name="class_drawbacks"></a>

The balancing of textual data using generated text is based on the simple concept of adding synthetic comments of the underrepresented class with the corresponding label to our data. Since we only took non-publishable comments as input for the generation models, our generated text is expected to imitate only those type of comments. This is an assumption that we make and since this is an essential point for our approach, we want to discuss it here in more detail. 

The comment data that we use for the text generation is probably noisy. We are not always able to explain why a comment has been banned from the comment section. Some comments might be labelled as non-publishable without any justification. For example, comments are marked as “not-published” before they were checked. There is a high probability that a comment, that was not manually verified so far, will be published after it was checked. Other comments might also be simply banned due to an erroneous assessment of the employee. There are legal and company-provided guidelines for the comment assessment. In some cases, the decision might differ due to the perception or interpretation of an employee that reads the comment. A comment that is at the boundary between being published and not being published, might be classified differently by different employees that check it. 

If this kind of noise is included in our data, then we reproduce the noise by generating comments based on that noisy text. By including the generated comments in our classification training phase, we increase the noisy data the classifier must deal with. Hence, it might become increasingly hard for the classifier to differentiate between both comment types.


## Classification Task - Limitations <a name="class_limitations"></a>

There are some limitations to our classification approach that we want to point out here. 

First, we used only the cleaned and tokenized text as an input for both our classifiers. Meta-information about the comments, like the number of words included or the date when a comment was created, were not taken into account. 

Second, due to the fact that the text generation takes a significant amount of time by using our limited computational resources, we were not able to use all our available comment data to train the models. We reduced the amount of comments in our training set to balance both classes dependent on the available amount of synthetic comments. 

Third, the usage of generated comments to balance our dataset inserts noise into our training data. The amount of noise is thereby highly dependent on the number of comments that are used as "original examples" for the generation models and also on the ratio of real and synthetic comments in the minority class. When only a very small number of text examples is available for the text generation, then the resulting synthetic comments might be too homogenous to serve as meaningful input for the classifier. Further, if there are only very few real comments and a lot of generated ones in the training data, then the amount of noise might confuse the classifier.


## Evaluation <a name="class_evaluation"></a>

For the evaluation of our classifiers across the different settings, we use three metrics: F1-Score, the AUC and accuracy (ACC). Additionally we measure the performance of our by examining the distribution of predictions. therefore, we compare the our results by the percentages of true/false positive and true/false negative class predictions. These values offer insights to the prediction behaviour of our models.

We are not able to evaluate the results including our third class “not sure”, because we have no ground truth available to check for the correctness of our predictions. The only possibility would be to look at those comments and try to assess whether they are indeed ambiguous.

The results are presented below. First the evlaution in terms of F1, AUC and ACC. We can observe that for both our baseline settings (Imbalanced and Undersample), the results are better for each metric in the Imbalanced case compared to the Undersampling setting. The results between the BOW and the RNN model are very similar for the Imbalanced case. In the Undersampling case we observe that the RNN model performs constantly worse than the BOW model, probably due to the lower number of comments in the training set. 

PLOT RESULTS BY SETTING (1)

For the two Oversampling settings, the results indicate a better performance for the models that were trained on generated text based on GPT-2 compared to the models from the GloVe setting. The results in both Oversampling cases are also constantly better compared to the Undersampling setting. However, if we compare our model performance from both balanced settings with generated comments with those from the Imbalanced case, we can observe that the results of the Imbalanced models are better for all of our metrics. This clearly indicates that the generated comments does not positively influence the performance of our classification models.

PLOT RESULTS BY SETTING (2)

This becomes even more clear if we look at the direct comparison of our metrics ordered by setting. We can observe there, that the blue bar, representing the imbalanced setting is the higest one for all our metrics. The AUC for our BOW-models are very similar across all four settings. But also there we observe the AUC of the Imbalanced setting to be slightly above those of the others. 

PLOT RESULTS BY METRIC

TABLE RESULTS

We further analyzed the distribution of class predictions and compared them to their corresponding true value. We use the confusion matrix of each model to extract the true/false-positives (TP, FP) and negatives (TN, FN).

Table Confusion Matrix

We used 50000 observations in our training set. In the chart below, the percentages for the four prediction cartegories are visualized. The "false" class predictions are represented in red and the "true" ones in blue. We can see that the Imbalanced setting again yielded at the most true values compared to the other scenarios. Regarding our business case, the class we want to identify most are the TN, i.e. the comments that are not publishable. Therefore, we want to minimize the numbers especially the amount of FP-predictions, i.e. the comments that are not publishable but were classified as publishable. THe percentage of correctly classified values (TP+TN) is higest in the Imbalanced setting. However, the percentage of FP and TN were minimal in the Undersampling setting. A severe drawback in the Undersampling case is that we observe a very high rate of FN, meaning that there are a lot of comments that are actually publishable but were clasified as not publishable by the classifier. This means that large amount of comments still needs to be checked manually.

There are some differences between the BOW and the RNN model, especially with regard to the Imbalanced setting. There, we observe that almost all comments are predicted to be publishable. This means that our RNN model is highly affected by the imbalanced distribution of comments. This situation is not observed in tha remaining (balanced) settings, although the performance of the Undersampling case are very low. This indicated that we were able to influence the predictions of our RNN model by using our generated comments. 

The predictions of the two Oversampling settings lie inbetween the two others. As already observed before, the setting including the GloVe-based generation of comments performed a bit worse than the GPT-2 scenario.

PLOT RESULTS RNN BOW PERCENTAGE FP FN TP TN (2)

## Including the "Not-Sure"-Category <a name="class_not_sure"></a>

As described in our business case at the beginning, we include a “not-sure”-category for comments, for which the classifier does not return predictions with a clear tendency towards a certain publishing status. These comments should be checked manually again by human employees to find a definitive decision. Therefore, the predictions of all models for the comments in the test data set where stored. To determine which comments should belong to the “not-sure”-class, it is necessary to specify a lower (e.g. 0.4) and an upper (e.g. 0.6) probability threshold. Comments, for which the predicted class probability lies withing these two thresholds, are then regarded as “not-sure”. 

Grafik not sure

The advantage of specifying an ambiguity range compared to setting a fixed probability threshold (of e.g. 0.5) is that comments for which the classifier could not find a clear tendency will be checked again by an employee. Assuming that the classification returns reliable probabilities, comments that lie around 0.5 can be perceived as difficult to assess. By doublechecking these comments manually by an employee, the final decision about whether a comment violates given legal and/or compliance restrictions, will be supported by human assessment. However, this assumes that our classifier is able to determine the publishing status with a high correctness level. This is necessary to ensure that comments that are outside the ambiguity range indeed belong to the corresponding class.

By manually checking comments with an ambiguous class prediction, we are able to evaluate the classification performance from another point of view. Since comments that are marked as “not-sure” are manually checked again, we can exclude them from the test set and repeat the evaluation with the remaining comments. The target is to assess the classifier performance only for comments where a clear tendency with respect to the publishing status was predicted. 

In the code below, we implemented the steps for including a “not-sure”-category. We therefore took only one example setting (Oversampling GPT-2), since it works equally for the other predictions. We first set the two thresholds and then we identified the indices of comments with a probability prediction within those two. Afterwards, we construct a new Data Frame without these identified comments. To compare the results with those including all comments, we calculate again F1-Score, AUC and accuracy as well as the resulting confusion matrix. 





```python
#setting thresholds for 'not-sure'-category
threshold1 = 0.4
threshold2 = 0.6

#find row indices to remove
drop_idx = pred_df['rnn_gpt2'][pred_df.rnn_gpt2.between(threshold1, threshold2)].index

# remove predictions that lie in between thresholds
pred_df_reduced = pred_df.drop(drop_idx)
pred_df_reduced = pred_df_reduced[['rnn_gpt2', 'true_value']]

#convert probabilities of reduced predictions dataframe to classes
pred_df_reduced_class = class_prediction(list(pred_df_reduced['rnn_gpt2']))
pred_df_reduced_class['true_value'] = list(pred_df_reduced.true_value)

#confusion matrix
confusion_matrix_rnn_gpt2_thresholds = conf_matrix(pred_df_reduced_class['class_pred'], pred_df_reduced_class['true_value'])
print(confusion_matrix_rnn_gpt2_thresholds)
```

We can see in the table below that the results were better in terms of the F1-Score and the Accuracy. However the AUC became significantly lower when excluding the "not-sure" comments. The reason for that becomes clear if we take a look at the two confusion tables. 

|          | including  all comments | excluding  "not sure"-comments |
|----------|:-----------------------:|:------------------------------:|
| **F1-Score** | 0.8758                  | 0.8892                         |
| **AUC**     | 0.6091                  | 0.5404                         |
| **ACC**      | 0.7852                  | 0.8052                         |

The big share of the TN (don't publish) observations were removed when including the "not-sure"-categroy. This has a negative impact on the AUC. Furthermore, that means that man comments that are predicted to be not pubishable have an ambiguous probability score. 

Cofusion Matrix Grafik


There are also severe differences between the impact of including a "not-sure"-class among the different models. The amount of comments that are labeled as "not sure" heavily depends on the model predictions. When looking at the distributions of the probability predictions of different models, we can recognize severe differences. The distributions represent the probability distributions for the BOW model in the Oversampling GPT-2 setting and the RNN model in the Imbalanced setting. We can clearly see that the GPT-2 setting BOW model predicts probabilities more distributed. Therefore there are more comments labelled as not sure. The RNN model from the Imbalanced setting on the other hand, predicts a probability close to 1 for most of the comments. 

Plots proba distributions

## Different Balancing Ratios for Generated Comments <a name="class_ratios"></a>

As already stated earlier, using generated comments to balance text data has the drawback of including noise into the oversampled class of text. In the previous settings where we used generated comments (Oversampling GloVe/GPT-2), we included that amount of synthetic comments of the underrepresented class (not pubished), that yielded approx. at a 50:50 – balancing ratio. To test whether the ratio has an impact on the results, we repeat the classification model training again with different ratios of published and not published comments. We try two different balancing ratios by iteratively reducing the number of synthetic comments. Afterwards, we compare the results of these two modified settings with the results of the imbalanced setting, where we include no generated comments and the 50:50-balancing ratio setting. The different ratios are described in the table below.


Table Ratios

The results show that the usage of synthetic comments negatively influences the model performance. We can see tha the results are best in the Imbalanced settings. The results improve if the ratio of comment classes gets closer to the imbalanced one. It seems that our generated comments insert too much noise and therefore causing a worse performance of the classification models.

PLOT RESULTS DIFFERENT RATIOS

## Conclusion and Discussion of Results <a name="class_conclusion"></a>

The evaluation of our results allowed us to assess the performance of our various models across settings. We aimed at improving text classification results by balancing text data with generated comments of the underrepresented class. The results do not support our hopethesis, that the intrusion of synthetic text improves our classification results. We observed that the utilization of genreated comments yielded at better results compared to the Undersample setting. However, the best results were produced by using the original imbalanced comment data. The results from the two Oversample settings were expected to be at least not worse than those from the Imbalanced.

There might be various reasons why we observed the final outcome. First, the intrusion of noise through using generated comments might overweight the potential advantages of training the models with a balanced dataset. Second, the change of the distribution of published and not published comments leads to higher numbers of minority class predictions. That suggests that the distribution of the training data might have an direct impact on the distribution of the predicted probabilities. Third, the quality of generated comments is surely an essential factor that influences classification training. During the generation phase, we produced comments of different quality levels (semantically and syntactically). Since we label all generated comments as not publishable, the classifier is possibly more likely to predict a higher amount of comments to be not publishable.

Further, there are several possible changes to our approach that might influence the results. First, there are other classification approaches that could be tested, like e.g. convolutional neural networks. Since we only used comparatively simple calssification methods, improving and finetunig the model architecture further might have animpact on the results. Second, the quality of genreated comments could be improved further. Comments that better represent the underrepresented class might reduce the amount of noise that is inserted in the training set. Third, the imbalancedness of the original comments might be not severe enough to make the intrusion of generated comments necessary. Maybe in cases where classes of comments are even more inequally distributed (e.g. 95:5) might lead to different results.


# References <a name="ref1"></a>

*   Allen, C. and Hospedales, T. (2019). Analogies Explained: Towards Understanding Word Embeddings. Proceedings of the 36th International Conference on Machine Learning, 97, pp.223-231.
*   Bahdanau, D., Cho, K. and Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. International Conference on Learning Representations, pp.1-9.
*   Georgakopoulos, S., Tasoulis, S., Vrahatis, A. and Plagianakos, V. (2018). Convolutional Neural Networks for Toxic Comment Classification. SETN '18: Proceedings of the 10th Hellenic Conference on Artificial Intelligence, pp.1-6.
*   Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), pp.1-2.
*   Kingma, D. and Ba, J. (2014). Adam: A Method for Stochastic Optimization. 3rd International Conference for Learning Representations.
*   Koehrsen, W. (2018). Recurrent Neural Networks by Example in Python. [online] Medium. Available at: https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470 [Accessed 4 Nov. 2019].
*   Liu, A., Ghosh, J. and Martin, C. (2007). Generative Oversampling for Mining Imbalanced Datasets. DMIN.
*   Luong, M., Pham, H. and Manning, C. (2015). Effective Approaches to Attention-based Neural Machine Translation. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp.1412–1421.
*   Papineni, K., Roukos, S., Ward, T. and Zhu, W. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp.311-318.
*   Schuster, M. and Paliwal, K. (1997). Bidirectional Recurrent Neural Networks. Transactions on Signal Processing, 45(11), pp.2673-2681.
*   Stamatatos, E. (2008). Author identification: Using text sampling to handle the class imbalance problem. Information Processing & Management, 44(2), pp.790-799.
*   van Aken, B., Risch, J., Krestel, R. and Loser, A. (2018). Challenges for Toxic Comment Classification: An In-Depth Error Analysis. ALW2: 2nd Workshop on Abusive Language Online to be held at EMNLP 2018, pp.1-8.
*   Bartoli, A., De Lorenzo, A., Medvet, E., & Tarlao, F. (2016, August). Your paper has been accepted, rejected, or whatever: Automatic generation of scientific paper reviews. In International Conference on Availability, Reliability, and Security (pp. 19-28). Springer, Cham.
*   Wu, S., & Dredze, M. (2019). Beto, bentz, becas: The surprising cross-lingual effectiveness of bert. arXiv preprint arXiv:1904.09077.
*   Reiter, E., & Dale, R. (1997). Building applied natural language generation systems. Natural Language Engineering, 3(1), 57-87.
*   anonymous authors (2020). From English to Foreign Languages: Transferring Pre-trained Language Models. ICLR 2020.
*   Dong, L., Huang, S., Wei, F., Lapata, M., Zhou, M., & Xu, K. (2017, April). Learning to generate product reviews from attributes. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers (pp. 623-632).
*   Xie, Z. (2017). Neural text generation: A practical guide. arXiv preprint arXiv:1711.09534.
*   Artetxe, M., Ruder, S., & Yogatama, D. (2019). On the cross-lingual transferability of monolingual representations. arXiv preprint arXiv:1910.11856.
*   Sriram, B., Fuhry, D., Demir, E., Ferhatosmanoglu, H., & Demirbas, M. (2010, July). Short text classification in twitter to improve information filtering. In Proceedings of the 33rd international ACM SIGIR conference on Research and development in information retrieval (pp. 841-842).
*   Bojanowski, P., Joulin, A., & Mikolov, T. (2015). Alternative structures for character-level RNNs. arXiv preprint arXiv:1511.06303.

## Picture References

*   Data Compression: https://developer.apple.com/documentation/compression
*   Byte Pair Encoding: https://www.thoughtvector.io/blog/subword-tokenization/
*   Word-Splitting: https://mc.ai/trends-in-input-representation-for-state-of-art-nlp-models-2019/


