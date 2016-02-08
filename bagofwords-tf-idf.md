---
title: Bag of Words - TF-IDF
layout: default
---

# Bag of Words - TF-IDF

[Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BoW) is a list of words with their wordcounts. Each row is a document, each column is a word and each cell is a wordcount. 

BoW is also a method for preparing text for input in a deep-learning net. For a given training corpus, it produces a list of words and their associated wordcounts. Each of the documents in the corpus is represented by columns of equal length. Those are wordcount vectors, an output stripped of context. 

Before they're fed to the neural net, each vector of wordcounts is normalized such that all elements of the vector add up to one. Thus, the frequencies of each word is effectively converted to represent the probabilities of those words' occurrence in the document. Probabilities that surpass certain levels will activate nodes in the net and influence the document's classification. 

### term-frequency-inverse document frequency

[Term-frequency-inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (tf-idf) is another way to judge the topic of an article by the words it contains. With tf-idf, words are given weight -- tf-idf measures relevance, not frequency. That is, wordcounts are replaced with tf-idf scores across the whole dataset. 

First, tf-idf measures the number of times that words appear in a given document (that's term frequency), but because words such as "and" or "the" appear frequently in all documents, those are systematically discounted. That's the inverse-document frequency part, which is intended to leave only the frequent AND distinctive words as markers. Tf-idf relevance is a normalized data format that also adds up to one. 

Those marker words are then fed to the neural net as features in order to determine the topic covered by the document that contains them. 

Setting up a BoW looks something like this: 

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

Click here to see [other BoW-based text-vectorization methods](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/deeplearning4j-nlp/src/main/java/org/deeplearning4j/bagofwords/vectorizer/BagOfWordsVectorizer.java).

BoW is different from [Word2vec](../word2vec.html), which we'll cover next. The main difference is that Word2vec produces one vector per word, whereas BoW produces one number (a wordcount). Word2vec is great for digging into documents and identifying content and subsets of content. Its vectors represent each word's context, the ngrams of which it is a part. BoW is good for classifying documents as a whole. 
