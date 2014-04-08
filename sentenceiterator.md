---
title: 
layout: default
---


# sentence iterator

A [sentence iterator](../doc/org/deeplearning4j/word2vec/sentenceiterator/SentenceIterator.html) is used in [Word2vec](../word2vec.html) as well as Bag Of Words approaches to text processing.

The [sentence iterator](../doc/org/deeplearning4j/word2vec/SentenceIterator.html) covers the concept of a document in text processing. A [sentence iterator](../doc/org/deeplearning4j/word2vec/sentenceiterator/SentenceIterator.html) handles feeding in bits of text to a neural network in the form of a vector.

Typically in natural language processing, a document or sentence is used to encapsulate a context which which an algorithm should learn.

A few examplels include wanting to do tweets vs full blown articles. The intention of the [sentence iterator](../doc/org/deeplearning4j/word2vec/sentenceiterator/SentenceIterator.html) is to handle figuring out how to divide up text in to processable bits. Note that the way the [sentence iterator](../doc/org/deeplearning4j/word2vec/SentenceIterator.html) works it is input agnostic. So bits of text (a document) can come from a file system, the twitter API, 
or hadoop.


Relative to how input is processed the output of a [sentence iterator](../doc/org/deeplearning4j/word2vec/sentenceiterator/SentenceIterator.html) will then go in to a [tokenizer](../org/deeplearning4j/word2vec/tokenizer/Tokenizer.html) for processing of individual tokens (typically words, but these could also be ngrams or other such units). The [tokenizer](../doc/org/deeplearning4j/word2vec/tokenizer/Tokenizer.html) is created on a per sentence basis by a [tokenizer factory](../doc/org/deeplearning4j/word2vec/tokenizer/TokenizerFactory.html). The [tokenizer factory](../doc/org/deeplearning4j/word2vec/tokenizer/TokenizerFactory.html) is 
what is passed in to a text processing vectorizer. 


Some typical examples are below:

            SentenceIterator iter = new LineSentenceIterator(new File("your file"));

This makes the assumption that each line in a file is a sentence.



You can also do list of strings as sentence as follows:

     Collection<String> sentences = ...;
     SentenceIterator iter = new CollectionSentenceIterator(sentences);

This will make the assumption that each string is a sentence(document). Keep in mind that this could be a list of tweets or articles. Both are applicable.


You can iterator over files as follows:

          
          SentenceIterator iter = new FileSentenceIterator(new File("your dir or file"));

This will then handle parsing the files line by line and return individual sentences on each one.


For anything complex, I would highly reccomend an actual machine learning level pipeline, this is represented by the [UimaSentenceIterator](../doc/org/deeplearning4j/text/sentenceiterator/UimaSentenceIterator.html).

The UimaSentenceIterator](../doc/org/deeplearning4j/text/sentenceiterator/UimaSentenceIterator.html) has built in capabilities for tokenization, part of speech tagging, lemmatization, among other things. The [UimaSentenceIterator](../doc/org/deeplearning4j/text/sentenceiterator/UimaSentenceIterator.html) iterates over a set of files and is capable of doing sentence segmentation for purposes of arbitrary documents. You can customize its behavior based on the AnalysisEngine passed in to it.

The AnalysisEngine is the [UIMA](http://uima.apache.org/) concept of a text processing pipeline. DeepLearning4j comes with some standard analysis engines for all of these common tasks

allowing you to customize what text is being passed in and what you want your concept of a sentence to be. The AnalysisEngines are thread safe versions of

the [opennlp](http://opennlp.apache.org/) pipelines. Included are also [cleartk](http://cleartk.googlecode.com/) based pipelines for handling common tasks.

For those of you who use UIMA or are curious, this uses the cleartk type system for tokenis, sentences, and other annotations within the type system.


Below is how to create a UimaSentenceItrator.

         SentenceIterator iter = UimaSentenceIterator.create("path/to/your/text/documents");

You can also instanitate directly with:

  SentenceIterator iter = new UimaSentenceIterator(path,AnalysisEngineFactory.createEngine(AnalysisEngineFactory.createEngineDescription(TokenizerAnnotator.getDescription(), SentenceAnnotator.getDescription())));


For those of you who are familiar with uima, this uses uimafit extensively for creation of analysis engines.



You can also create your own custom sentence iterators by extending [sentence iterator](../doc/org/deeplearning4j/word2vec/sentenceiterator/BaseSentenceIterator.html)

