<!--

---
layout: default
title: Sentiment Analysis With Word2vec
---

# Sentiment Analysis With Word2vec

Contents

* <a href="#bow">Bag of Words v. Word2vec</a>
* <a href="#count">Beyond Word Counts</a>
* <a href="#sentiment">Sentiment Analysis</a>
* <a href="#softmax">Softmax Logistic Regression</a>
* <a href="#log">Log Likelihood</a>
* <a href="#code">**Just Give Me The Code**</a>
* <a href="#resource">Other Resources</a>

This tutorial covers sentiment analysis with [Word2vec](../word2vec.html) and logistic regression. It is written for programmers, but assumes knowledge of only basic mathematical concepts. Its purpose is to demonstrate how Word2vec can be used for opinion mining on text in the wild. 

Sentiment has obvious applications for market research, business intelligence, product development, reputation management, political campaigns and sociological studies. 

Broadly speaking, sentiment analysis has three stages: tokenization, feature extraction and classification. The first divides a document up into words, the second creates representations of words or documents, the third learns to bucket them by type. For this tutorial, we are interested in feature extraction and classification on the document level. 

## <a name="bow">Bag of Words vs. Word2vec</a>

Typically, people rely on word count to create vectors representing documents, and measure their similarities and differences by the frequency with which terms appear. To illustrate this word count approach with a toy example, let’s say we care about two words, Moscow and Beijing. 

First, we count the number of times those words appear in two documents. Let’s say the x axis represents Moscow and the y axis Beijing. If Moscow appears once in document one and five times in document two, while Beijing appears three times in the first and twice in the second, then we have our two vectors: *doc1 = (1,3)* and *doc2 = (5,2)*.

![Alt text](../img/sentiment_analysis_vectors.png)

With Bag of Words, you can add as many dimensions as there are unique words in your documents, placing vectors in an n-dimensional space. Here's an example of a vector wending its way through three dimensions:

![Alt text](../img/3d_vector.png)

To imagine n dimensions, just keep turning sharply through the wormhole to the right. ;)

You would measure the difference in the angle between those vectors to arrive at one expression of similarity. 

There are other nuances to wordcount, such as [term frequency-inverse document frequency (TF-IDF)](https://en.wikipedia.org/wiki/Tf–idf), but that is not our focus here.

Bag of Words is sparse. It creates a vector with one component for every word in the corpus, even though for any given document, many of those components will be 0. ([Shakespeare](http://www.opensourceshakespeare.org/stats/) used about 30,000 unique words, almost half of which appeared only once.)

Word2vec imitates sparse in that words that don't occur don't get counted, but a 500-dimensional neural word embedding is dense. You could argue that there's less noise because it's dense, you don't have to handle the overhead of vectors large enough for all the words all the time. 

## <a name="count">Beyond Word Count</a>

Bag of Words has a weakness, which is that it doesn’t necessarily know whether you’re drinking a Coca-Cola or investing in Coca-Cola or worse yet, a millionaire investing in Coca-Cola who’s drinking a Coca-Cola; i.e. it gathers information about usage by measuring frequency, and doesn’t understand context. 

Or to return to the example above, mere word count wouldn't necessarily tell us if a document was about an alliance between Moscow and Beijing and conflict with a third party, or alliances with third parties and conflict between Moscow and Beijing. 

But Word2vec *does* grasp context, because the algorithm learns to reconstruct the context surrounding each word. (Word2vec has two forms, one of which infers a target word from its context, while the other infers the context from the target word. We use the latter.)

![Alt text](../img/word2vec_diagrams.png)

So the vectors produced by Word2vec are not populated with word counts. They are neural word embeddings that allow Word2vec to predict, given a certain word, what the most likely words around it are. 

Creating a representation of those words for sentiment analysis is as simple as adding together the feature vectors for each word in the document and dividing that vector by the number of word vectors extracted from the document. 

![Alt text](../img/avg_word_vector.png)

This document vector (not to be confused with doc2vec) can then be compared to vectors representing documents in, say, a labeled set. 

## <a name="sentiment">Sentiment Analysis </a>

Sentiment analysis goes by many names: opinion extraction, opinion mining, sentiment mining and subjectivity analysis. Whatever the name, it is a kind of text classification: Is the movie review positive or negative? 

It can also help answer the secondary question of how people felt about specific aspects of a thing. A printer, for example, might be judged according to its speed, reliability, ease of use, tech support and connectivity. 

In addition, the stream of comments harvested from social media can be used for tasks as diverse as stock market or election predictions and brand monitoring. In the code example below, we will show you how to analyze Tweets with regard to company names. The analysis is conducted on the sentence level with regard to a supervised training set, rather than using a lexicon of sentiment-labeled words. 

For neural networks to learn sentiment, you need a labeled dataset to conduct supervised learning; i.e. you must have a set of documents or words that humans have associated with emotional signals, be they as simple as *positive* and *negative*, or as nuanced as frustration, anger, delight, satisfaction and lacadaisical whimsicality. (See [Scherer](http://www.affective-sciences.org/user/scherer)'s [Typology of Affective States](http://www.amazon.com/Handbook-Affective-Sciences-Science/dp/0195126017) for a more rigorous categorization.)

![Alt text](../img/scherer.png)

So the first step is to pick the categories you care about. The second is to create a dataset in which the examples have been tagged with those labels. (Mechanical Turk is useful here...) If you don't mind using someone else's corpus and categories, you can download this set of [Tweets](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) labeled by their positive and negative sentiment, as we show below.

We use this code to parse a CSV file of Tweets:

        log.info("Parse CSV file from s3 bucket");
        CsvSentencePreprocessor csvSentencePreprocessor = new CsvSentencePreprocessor();
        S3SentenceIterator it = new S3SentenceIterator(csvSentencePreprocessor, "sentiment140twitter", "sentiment140_train.csv");

Once you have a corpus, you feed its words into Word2vec to generate feature vectors. For each Tweet, those feature vectors are added and the aggregate word vector is divided by the number of words. That "average word vector" serves as the input for logistic regression. When you have the corpus labeled by sentiment, you can conduct supervised training to obtain a model that can predict those labels. 

## <a name="softmax">Softmax Logistic Regression</a>

[Logistic regression](http://gormanalysis.com/logistic-regression-fundamentals/), despite its misleading name, classifies things. The simplest form of logistic regression is binary, predicting categories such as *spam* or *not_spam*. It does so by mapping input to a sigmoid space where large numbers asymptotically approach 1, and small ones approach zero.

![Alt text](../img/sigmoid2.png)

The equation that maps continuous input to sigmoid looks like this:

![Alt text](../img/logistic_regression2.png)

With the slider in your mind, imagine that as *t* grows larger, so does *e*'s negative exponent, which means that *e^-t* approaches zero and the formula leaves us with 1; and vice versa with increasingly large negative *t*'s, which leave us with 0.

Given one or more input variables, it estimates the probability that the input belongs to one category or another. Each variable of the input is a component of a vector, one of the feature vectors mentioned above. Imagine a feature vector of Betas, the weights that modify an input vector of x's. 

![Alt text](../img/logistic_regression3.png)

A more complex form that buckets input into more than two categories is called multinomial logistic regression, or *softmax*. That’s what we’ll be using here. 

Softmax dot multiplies an input vector by a weight vector, using separate weight vectors for each respective output. It then squashes the results into a narrow range, and assumes the category whose weight vector produces the highest result is the correct classification. Thus the *max*. 

With supervised learning, we adjust the weight vectors of softmax until they properly categorize input based on the human-labelled training set. Those adjustments minimize a cost function using something called *negative log likelihood*. 

## <a name="log">Log Likelihood</a>

Let's talk about likelihoods and logs. 

The probability of an event is calculated based on input variables and the weights used to adjust them. It's easy to understand that you feed observations into a model to come up with a prediction. 

But what if the predictions are given and the weights are unknown? *Likelihood* is the probability that certain weights will occur, given the outcomes you know, which is particularly useful when you have a training set where the labels and observations are fixed, and the weights have yet to be determined. 

The optimization algorithm used here attempts to find the weights with the maximum likelihood, given the labels and input. It estimates the likelihood of a model’s parameters (the weight vectors of softmax, in this case), given the output, or the ground-truth classifications of the training set. It is an inversion of the probability function of output given certain parameters. 

By maximizing the likelihood of the parameters with regard to the ground-truth labels, we train a model that can classify unseen data better. 

Taking the logarithm of input *x* -- i.e. finding the exponent of 10 that produces *x* -- is a useful way to map very low likelihoods into a narrower and more manageable space. For example, *log(0.0001)* can be expressed as -4. Using log likelihoods prevents [arithmetic underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow), which is a risk when you deal with very low probabilities.

We then flip the log likelihoods to the negative, because most optimization algorithms attempt to *minimize* a cost or error function, and negative log likelihood plays nice with a more general framework for optimization, like Deeplearning4j.

Once the weights can no longer be adjusted to reduce error — i.e. once the likelihood of the parameters given the output reaches its peak — the model can be used to categorize input for which no labels exist, inferring sentiment expressed by text in the wild. 

## <a name="code">Just Give Me the Code</a>

Below is the `main` method of our program. The comments give a high-level description of the steps involved. Here you can find our full implementation of the methods used in [sentiment analysis with Word2vec](https://github.com/deeplearning4j/twitter_sentiment_analysis).

    public static void main(String args[]) throws Exception {
        String s3BucketName = "sentiment140twitter";
        
        // Separate the test set from the training set... 
        
        // Full set of data, takes ~20 mins to train
        // String trainDataFileName = "sentiment140_train.csv";
        
        // Smaller dataset saves training time, lessens accuracy
        String trainDataFileName = "sentiment140_train_sample50th.csv";
        String testDataFileName = "sentiment140_test.csv";
        
        // declare how many features we want per word vector
        int vectorLength = 200;
             
        RunAnalysis runAnalysis = new RunAnalysis(s3BucketName, trainDataFileName, testDataFileName, vectorLength);
        
        //creates the feature vector for each word
        runAnalysis.runWord2Vec();
        
        //computes average word vector per tweet
        Pair<List<INDArray>, List<INDArray>> targetFeaturePair = runAnalysis.computeAvgWordVector();
        
        //supervised training of word vectors against sentiment labels
        MultiLayerNetwork model = runAnalysis.trainSentimentClassifier(targetFeaturePair);
      }
    }

## <a name="resource">Other Resources</a>

* [Natural Language Processing on Coursera](https://class.coursera.org/nlp/) (Chris Manning and Dan Jurafsky)
* [Exploiting Similarities among Languages for Machine Translation](http://arxiv.org/pdf/1309.4168.pdf); Mikolov et al
* [Twitter Sentiment Classification using Distant Supervision](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf); Alec Go et al
* [Thumbs up? Sentiment classification using machine learning techniques](http://www.cs.cornell.edu/home/llee/papers/sentiment.home.html); Bo Pang et al
* [A Beginner's Guide to Word2Vec](../word2vec.html)
