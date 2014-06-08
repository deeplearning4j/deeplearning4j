---
title: 
layout: default
---

*previous* - [bag of words - tf-idf](../bagofwords-tf-idf.html)
# word2vec

One of deep learning's chief applications is in textual analysis, and at the heart of text analysis is [Word2vec](https://code.google.com/p/word2vec/). Word2vec is a neural network that does not implement deep learning, but is crucial to getting input in a numerical form that deep-learning nets can ingest -- the [vector](https://www.khanacademy.org/math/linear-algebra/vectors_and_spaces/vectors/v/vector-introduction-linear-algebra). 

Word2vec creates features without human intervention, and some of those features include the context of individual words; that is, it retains context in the form of multiword windows. In deep learning, the meaning of a word is essentially the words that surround it. Given enough data, usage and context, Word2vec can make highly accurate guesses as to a word’s meaning based on its past appearances. 

The output of the Word2vec neural net is a vocabulary with a vector attached to it, which can be fed into a deep-learning net for classification/labeling. 

There is also a [skip gram representation](http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf) which is used in the dl4j implementation. This has proven to be more accurate due to the more generalizable contexts generated. 

Broadly speaking, we measure words' proximity to each other through their [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), which gauges the similarity between two word vectors. A perfect 90 degree angle represents identity; i.e. France equals France, while Spain has a cosine distance of 0.678515 from France, the highest of any other country.

Here's a graph of words associated with "China" using Word2vec:

![Alt text](../img/word2vec.png)

The other method of preparing text for input to a deep-learning net is called [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BoW). BoW produces a vocabulary with word counts associated to each element of the text. Its output is a wordcount vector. That said, it does not retain context, and therefore is not useful in a granular analysis of those words' meaning. 

## training

Word2Vec trains on raw text. It then records the context, or usage, of each word encoded as word vectors. After training, it's used as lookup table for composition of windows of training text for various tasks in natural-language processing.

Assuming a list of sentences, it's used for lemmatization like this:

<script src="http://gist-it.appspot.com/https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/word2vec/MovingWindowExample.java?slice=44:75"></script>

From there, Word2vec will do automatic multithreaded training based on your sentence data. After that step, you'll' want to save word2vec like this:

       	 SerializationUtils.saveObject(vec, new File("mypath"));

This will save word2vec to mypath.

You can reload it into memory like this:
        
        Word2Vec vec = SerializationUtils.readObject(new File("mypath"));

From there, you can use Word2vec as a lookup table in the following way:
              
        DoubleMatrix wordVector = vec.getWordVectorMatrix("myword");

        double[] wordVector = vec.getWordVector("myword");

If the word isn't in the vocabulary, Word2vec returns zeros and nothing more.

### windows

Word2Vec works with other neural networks by facilitating the moving-window model for training on word occurrences. There are two ways to get windows for text:

      List<Window> windows = Windows.windows("some text");

This will select moving windows of five tokens from the text (each member of a window is a token).

You also may want to use your own custom tokenizer like this:

      TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

      List<Window> windows = Windows.windows("text",tokenizerFactory);

This will create a tokenizer for the text, and moving windows based on the tokenizer.

      List<Window> windows = Windows.windows("text",tokenizerFactory);

This will create a tokenizer for the text and create moving windows based on that.

Notably, you can also specify the window size like so:

      TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

      List<Window> windows = Windows.windows("text",tokenizerFactory,windowSize);

Training word sequence models is done through optimization with the [Viterbi algorithm](../doc/org/deeplearning4j/word2vec/viterbi/Viterbi.html).

The general idea is that you train moving windows with Word2vec and classify individual windows (with a focus word) with certain labels. This could be done for part-of-speech tagging, semantic-role labeling, named-entity recognition and other tasks.

Viterbi calculates the most likely sequence of events (labels) given a transition matrix (the probability of going from one state to another). Here's an example snippet for setup:

<script src="http://gist-it.appspot.com/https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/word2vec/MovingWindowExample.java?slice=112:121"></script>

From there, each line will be handled something like this:

        <ORGANIZATION> IBM </ORGANIZATION> invented a question-answering robot called <ROBOT>Watson</ROBOT>.

Given a set of text, Windows.windows automatically infers labels from bracketed capitalized text.

If you do this:

        String label = window.getLabel();

on anything containing that window, it will automatically contain that label. This is used in bootstrapping a prior distribution over the set of labels in a training corpus.

The following code saves your Viterbi implementation for later use:
       
        SerializationUtils.saveObject(viterbi, new File("mypath"));

That's pretty much it. If you need help, [drop us a line](http://www.skymind.io/contact.html). 