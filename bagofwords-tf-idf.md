---
title: Bag of Words - TF-IDF
layout: default
---

# Bag of Words & TF-IDF 

[Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BoW) is an algorithm that counts how many times a word appears in a document. Those word counts allow us to compare documents and gauge their similarities for applications like search, document classification and topic modeling. BoW is a method for preparing text for input in a deep-learning net. 

BoW lists words with their word counts per document. In the table where the words and documents effectively become vectors are stored, each row is a word, each column is a document and each cell is a word count. Each of the documents in the corpus is represented by columns of equal length. Those are wordcount vectors, an output stripped of context. 

![Alt text](./img/wordcount-table.png) 

Before they're fed to the neural net, each vector of wordcounts is normalized such that all elements of the vector add up to one. Thus, the frequencies of each word is effectively converted to represent the probabilities of those words' occurrence in the document. Probabilities that surpass certain levels will activate nodes in the net and influence the document's classification. 

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH MACHINE LEARNING</a>
</p>

### Term Frequency-Inverse Document Frequency (TF-IDF)

[Term-frequency-inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (TF-IDF) is another way to judge the topic of an article by the words it contains. With TF-IDF, words are given weight -- TF-IDF measures relevance, not frequency. That is, wordcounts are replaced with TF-IDF scores across the whole dataset. 

First, TF-IDF measures the number of times that words appear in a given document (that's term frequency). But because words such as "and" or "the" appear frequently in all documents, those are systematically discounted. That's the inverse-document frequency part. The more documents a word appears in, the less valuable that word is as a signal. That's intended to leave only the frequent AND distinctive words as markers. Each word's TF-IDF relevance is a normalized data format that also adds up to one. 

![Alt text](./img/tfidf.png) 

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



BoW is different from [Word2vec](./word2vec.html), which we'll cover next. The main difference is that Word2vec produces one vector per word, whereas BoW produces one number (a wordcount). Word2vec is great for digging into documents and identifying content and subsets of content. Its vectors represent each word's context, the ngrams of which it is a part. BoW is good for classifying documents as a whole. 

## <a name="intro">More Machine Learning Tutorials</a>

* [Deep Reinforcement Learning](./deepreinforcementlearning.html)
* [Deep Convolutional Networks](./convolutionalnets.html)
* [Recurrent Networks and LSTMs](./lstm.html)
* [Multilayer Perceptron (MLPs) for Classification](./multilayerperceptron.html)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network.html)
* [Symbolic Reasoning & Deep Learning](./symbolicreasoning.html)
* [Using Graph Data with Deep Learning](./graphdata.html)
* [AI vs. Machine Learning vs. Deep Learning](./ai-machinelearning-deeplearning.html)
* [Markov Chain Monte Carlo & Machine Learning](/markovchainmontecarlo.html)
* [MNIST for Beginners](./mnist-for-beginners.html)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine.html)
* [Eigenvectors, PCA, Covariance and Entropy](./eigenvector.html)
* [Glossary of Deep-Learning and Neural-Net Terms](./glossary.html)
* [Word2vec and Natural-Language Processing](./word2vec.html)
* [Deeplearning4j Examples via Quickstart](./quickstart.html)
* [Neural Networks Demystified](https://www.youtube.com/watch?v=bxe2T-V8XRs) (A seven-video series)
* [Inference: Machine Learning Model Server](./modelserver.html)
