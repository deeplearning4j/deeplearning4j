---
title: Bag of Words - TF-IDF
layout: default
---

# Bag of Words & TF-IDF 

[Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BoW) is an algorithm that counts how many times a word appears in a document. Those word counts allow us to compare documents and gauge their similarities for applications like search, document classification and topic modeling. BoW is a method for preparing text for input in a deep-learning net. 

BoW lists words with their word counts per document. In the table where the words and documents effectively become vectors are stored, each row is a word, each column is a document and each cell is a word count. Each of the documents in the corpus is represented by columns of equal length. Those are wordcount vectors, an output stripped of context. 

IMAGE TK

Before they're fed to the neural net, each vector of wordcounts is normalized such that all elements of the vector add up to one. Thus, the frequencies of each word is effectively converted to represent the probabilities of those words' occurrence in the document. Probabilities that surpass certain levels will activate nodes in the net and influence the document's classification. 

### Term Frequency-Inverse Document Frequency (TF-IDF)

[Term-frequency-inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (TF-IDF) is another way to judge the topic of an article by the words it contains. With TF-IDF, words are given weight -- TF-IDF measures relevance, not frequency. That is, wordcounts are replaced with TF-IDF scores across the whole dataset. 

First, TF-IDF measures the number of times that words appear in a given document (that's term frequency). But because words such as "and" or "the" appear frequently in all documents, those are systematically discounted. That's the inverse-document frequency part. The more documents a word appears in, the less valuable that word is as a signal. That's intended to leave only the frequent AND distinctive words as markers. TF-IDF relevance is a normalized data format that also adds up to one. 

IMAGE TK

Those marker words are then fed to the neural net as features in order to determine the topic covered by the document that contains them.

Setting up a BoW looks something like this: 

``` java
    public class BagOfWordsVectorizer extends BaseTextVectorizer {
      public BagOfWordsVectorizer(){}
      protected BagOfWordsVectorizer(VocabCache cache,
             TokenizerFactory tokenizerFactory,
             List<String> stopWords,
             int minWordFrequency,
             DocumentIterator docIter,
             SentenceIterator sentenceIterator,
             List<String> labels,
             InvertedIndex index,
             int batchSize,
             double sample,
             boolean stem,
             boolean cleanup) {
          super(cache, tokenizerFactory, stopWords, minWordFrequency, docIter, sentenceIterator,
              labels,index,batchSize,sample,stem,cleanup);
    }
```

While simple, TF-IDF is incredibly powerful, and contributes to such ubiquitous and useful tools as Google search. 

Click here to see [other BoW-based text-vectorization methods](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/deeplearning4j-nlp/src/main/java/org/deeplearning4j/bagofwords/vectorizer/BagOfWordsVectorizer.java).

BoW is different from [Word2vec](../word2vec.html), which we'll cover next. The main difference is that Word2vec produces one vector per word, whereas BoW produces one number (a wordcount). Word2vec is great for digging into documents and identifying content and subsets of content. Its vectors represent each word's context, the ngrams of which it is a part. BoW is good for classifying documents as a whole. 

### <a name="beginner">Other Deeplearning4j Tutorials</a>
* [Word2vec](../word2vec)
* [Introduction to Neural Networks](../neuralnet-overview)
* [Restricted Boltzmann Machines](../restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](../eigenvector)
* [LSTMs and Recurrent Networks](../lstm)
* [Neural Networks and Regression](../linear-regression)
* [Convolutional Networks for Images](../convolutionalnets)
