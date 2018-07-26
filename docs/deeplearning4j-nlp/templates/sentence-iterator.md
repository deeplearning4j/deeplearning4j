---
title: Sentence Iteration
short_title: Sentence Iteration
description: Iteration of words, documents, and sentences for language processing in DL4J.
category: Language Processing
weight: 10
---

## Sentence iterator

A [sentence iterator](./doc/org/deeplearning4j/word2vec/sentenceiterator/SentenceIterator.html) is used in both [Word2vec](./word2vec.html) and [Bag of Words](./bagofwords-tf-idf.html).

It feeds bits of text into a neural network in the form of vectors, and also covers the concept of documents in text processing.

In natural-language processing, a document or sentence is typically used to encapsulate a context which an algorithm should learn.

A few examples include analyzing Tweets and full-blown news articles. The purpose of the [sentence iterator](./doc/org/deeplearning4j/word2vec/sentenceiterator/SentenceIterator.html) is to divide text into processable bits. Note the sentence iterator is input agnostic. So bits of text (a document) can come from a file system, the Twitter API or Hadoop.

Depending on how input is processed, the output of a sentence iterator will then be passed to a [tokenizer](./org/deeplearning4j/word2vec/tokenizer/Tokenizer.html) for the processing of individual tokens, which are usually words, but could also be ngrams, skipgrams or other units. The tokenizer is created on a per-sentence basis by a [tokenizer factory](./doc/org/deeplearning4j/word2vec/tokenizer/TokenizerFactory.html). The tokenizer factory is what is passed into a text-processing vectorizer. 

Some typical examples are below:

         SentenceIterator iter = new LineSentenceIterator(new File("your file"));

This assumes that each line in a file is a sentence.

You can also do list of strings as sentence as follows:

	     Collection<String> sentences = ...;
	     SentenceIterator iter = new CollectionSentenceIterator(sentences);

This will assume that each string is a sentence (document). Remember this could be a list of Tweets or articles -- both are applicable.

You can iterate over files as follows:
          
          SentenceIterator iter = new FileSentenceIterator(new File("your dir or file"));

This will parse the files line by line and return individual sentences on each one.

For anything complex, we recommend an actual machine-learning level pipeline, represented by the [UimaSentenceIterator](./doc/org/deeplearning4j/text/sentenceiterator/UimaSentenceIterator.html).

The UimaSentenceIterator is capable of tokenization, part-of-speech tagging and lemmatization, among other things. The UimaSentenceIterator iterates over a set of files and can segment sentences. You can customize its behavior based on the AnalysisEngine passed into it.

The AnalysisEngine is the [UIMA](http://uima.apache.org/) concept of a text-processing pipeline. DeepLearning4j comes with standard analysis engines for all of these common tasks, allowing you to customize which text is being passed in and how you define sentences. The AnalysisEngines are thread-safe versions of the [opennlp](http://opennlp.apache.org/) pipelines. We also include [cleartk](http://cleartk.googlecode.com/)-based pipelines for handling common tasks.

For those using UIMA or curious about it, this employs the cleartk type system for tokens, sentences, and other annotations within the type system.

Here's how to create a UimaSentenceItrator.

            SentenceIterator iter = UimaSentenceIterator.create("path/to/your/text/documents");

You can also instantiate directly:

			SentenceIterator iter = new UimaSentenceIterator(path,AnalysisEngineFactory.createEngine(AnalysisEngineFactory.createEngineDescription(TokenizerAnnotator.getDescription(), SentenceAnnotator.getDescription())));

For those familiar with Uima, this uses Uimafit extensively to create analysis engines. You can also create custom sentence iterators by extending SentenceIterator.