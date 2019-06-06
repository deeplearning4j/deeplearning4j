---
title: Deeplearning4j's NLP Functionality
short_title: Overview
description: Overview of language processing in DL4J
category: Language Processing
weight: 0
---

## Deeplearning4j's NLP Functionality

Although not designed to be comparable to tools such as Stanford CoreNLP or NLTK, deepLearning4J does include some core text processing tools that are described here.

Deeplearning4j's NLP relies on [ClearTK](https://cleartk.github.io/cleartk/), an open-source machine learning and natural language processing framework for the Apache [Unstructured Information Management Architecture](https://uima.apache.org/), or UIMA. UIMA enables us to perform language identification, language-specific segmentation, sentence boundary detection and entity detection (proper nouns: persons, corporations, places and things).

### SentenceIterator

There are several steps involved in processing natural language. The first is to iterate over your corpus to create a list of documents, which can be as short as a tweet, or as long as a newspaper article. This is performed by a SentenceIterator, which will appear like this:

<script src="https://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec/Word2VecRawTextExample.java?slice=33:41"></script>

The SentenceIterator encapsulates a corpus or text, organizing it, say, as one Tweet per line. It is responsible for feeding text piece by piece into your natural language processor. The SentenceIterator is not analogous to a similarly named class, the DatasetIterator, which creates a dataset for training a neural net. Instead it creates a collection of strings by segmenting a corpus.

### Tokenizer

A Tokenizer further segments the text at the level of single words, also alternatively as n-grams. ClearTK contains the underlying tokenizers, such as parts of speech (PoS) and parse trees, which allow for both dependency and constituency parsing, like that employed by a recursive neural tensor network (RNTN).

A Tokenizer is created and wrapped by a [TokenizerFactory](https://github.com/deeplearning4j/deeplearning4j/blob/6f027fd5075e3e76a38123ae5e28c00c17db4361/deeplearning4j-scaleout/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/tokenization/tokenizerfactory/UimaTokenizerFactory.java). The default tokens are words separated by spaces. The tokenization process also involves some machine learning to differentiate between ambibuous symbols like . which end sentences and also abbreviate words such as Mr. and vs.

Both Tokenizers and SentenceIterators work with Preprocessors to deal with anomalies in messy text like Unicode, and to render such text, say, as lowercase characters uniformly.

<script src="https://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec/Word2VecRawTextExample.java?slice=43:57"></script>



### Vocab

Each document has to be tokenized to create a vocab, the set of words that matter for that document or corpus. Those words are stored in the vocab cache, which contains statistics about a subset of words counted in the document, the words that "matter". The line separating significant and insignifant words is mobile, but the basic idea of distinguishing between the two groups is that words occurring only once (or less than, say, five times) are hard to learn and their presence represents unhelpful noise.

The vocab cache stores metadata for methods such as Word2vec and Bag of Words, which treat words in radically different ways. Word2vec creates representations of words, or neural word embeddings, in the form of vectors that are hundreds of coefficients long. Those coefficients help neural nets predict the likelihood of a word appearing in any given context; for example, after another word. Here's Word2vec, configured:

<script src="https://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/word2vec/Word2VecRawTextExample.java"></script>

Once you obtain word vectors, you can feed them into a deep net for classification, prediction, sentiment analysis and the like.