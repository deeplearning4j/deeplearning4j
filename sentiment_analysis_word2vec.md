---
layout: default
---

# Sentiment Analysis With Word2vec

Contents

* <a href="#vectors">Vectors</a>
* <a href="#trig">Trigonometric Functions</a>
* <a href="#bow">Bag of Words v. Word2vec</a>
* <a href="#count">Beyond Word Counts</a>
* <a href="#sentiment">Sentiment Analysis</a>
* <a href="#softmax">Softmax Logistic Regression</a>
* <a href="#code">Just Give Me The Code</a>
* <a href="#resource">Other Resources</a>

This tutorial covers sentiment analysis with Word2vec and logistic regression. It is written for programmers, but assumes knowledge of only basic mathematical concepts. Its purpose is to demonstrate how word2vec can be used for opinion mining on text in the wild. 

## Vectors

Word2vec learns to represent words as vectors.

A vector is a data structure with at least two components, as opposed to a *scalar*, which has just one. For example, a vector can represent velocity, an idea that combines speed and direction: *wind velocity* = (50mph, 35 North East). A scalar, on the other hand, can represent something with one value like temperature or height: 50 degrees Celsius, 180 centimeters.

Therefore, we can represent two-dimensional vectors as arrows on an x-y graph, with the coordinates x and y each representing one of the vector's values. 

PICTURE OF VECTOR

These vectors relate mathematically, and therefore similarities between them (and therefore between anything you can vectorize, including words) can be established. One way to do that is with a dot product, or a form of vector multiplication. 

PICTURE OF TWO VECTORS

As you can see, these vectors differ from one another in both their length, or magnitude, and their angle, or direction. The angle is what concerns us here, and two find its difference, we need to know the formula for vector dot multiplication (multiplying two vectors to produce a single, scalar value).

VECTOR DOT PRODUCT FORMULA - COLOR CODED
FORMULA REWRITTEN

TKTKTK Explain the terms….

In Java, you can think of the same formula like this:

    public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }   
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

## Trigonometric Functions

Let's do a quick trig review. 

Trigonometric functions like *sine*, *cosine* and *tangent* are ratios that use the lengths of a side of a right triangle (opposite, adjacent and hypotenuse) to compute the shape’s angles. By feeding the sides into ratios like 

INSERT: opposite over hypotenuse, adjacent over hypotenuse, opposite over adjacent.
NAME  

SOH-CAH-TOA

we can also know the angles at which those sides intersect. 

Cosine is the angle attached to the origin, which makes it useful here. Differences between word vectors, as they swing around the origin like the arms of a clock, can be thought of as differences in degrees. (We normalize them so they come out as percentages, where 1 means that two vectors are equal, and 0 means they are perpendicular; i.e. bear no relation to each other.)

## Bag of Words vs. Word2vec

We are interested in sentiment analysis and classification on the document level. Typically, people rely on wordcount to create vectors representing documents, and to measure their similarities and differences. To illustrate this with a toy example, let’s say we just cared about two words, Moscow and Beijing.

First, we count the number of times those words appear in two documents. Let’s say the x axis represents Moscow and the y axis Beijing. If Moscow appears once in document one and five times in document two, while Beijing appears three times in the first and twice in the second, then we have our two vectors: doc1 = (1,3) and doc2 = (5,2).

PICTURE VECTORS

With Bag of Words, you can add as many dimensions as there are unique words in your documents, placing vectors in an n-dimensional space. And you would measure the difference in the angle between those vectors to arrive at one expression of similarity. 

There are other nuances to wordcount, such as term frequency-inverse document frequency (TF-IDF), but since that is not our focus in this tutorial, we will simply link to the Wikipedia page.

## Beyond Word Count

There’s a problem with Bag of Words, which is that it doesn’t necessarily know whether you’re drinking a Coca-Cola or investing in Coca-Cola or worse yet, an investor who’s drinking a Coke; i.e. it doesn’t understand context. But Word2vec does, because the algorithm learns to reconstruct the context surrounding each word. 

So the vectors produced by Word2vec are not populated with word counts. They are neural word embeddings that allow Word2vec to predict, given a certain word, what the most likely surrounding words are.

Creating a document-vector representation for sentiment analysis is as simple as adding together the feature vectors for each word in the document and dividing that vector by the number of word vectors extracted from the document. 

EQUATION HERE

This document vector (not to be confused with doc2vec) can then be compared to vectors representing documents in, say, a labeled set.

## Sentiment Analysis 

For neural networks to learn sentiment, you need a labeled dataset to conduct supervised learning; i.e. you must have a set of documents or words that humans have associated with emotional signals, be they as simple as *positive*, *negative* and *neutral*, or as nuanced as frustration, anger, delight and satisfaction.
So the first step is to pick the categories you care about. 

Logistic regression will learn to classify documents, and then to manually associate those labels with a dataset for training. (Mechanical Turk is useful here...)

Word2vec can be used in place of typical Bag-of-Words or TF-IDF techniques to generate a vector representation of a document. Basically, you add up the vectors of all the words in the document, and then divide that aggregate vector by the number of words. That document vector serves as the input for logistic regression. 

CODE HERE

## Softmax Logistic Regression

Logistic regression, despite its misleading name, classifies things. Given one or more input variables, it estimates the probability that the input belongs to one category or another. Each variable of the input is a component of a vector, a feature vector fed into the classifier. 

The simplest form of logistic regression is binary, predicting categories such as *spam* or *not_spam*. A more complex form that buckets input in more than two categories is called multinomial logistic regression, or softmax. That’s what we’ll be using here. 

Softmax dot multiplies an input vector by a weight vector, using separate weight vectors for each respective output. It then squashes the results into a narrow range, and assumes the category whose weight vector produces the highest result is the correct classification. Thus the *max*.

With supervised learning, we adjust the weight vectors of softmax until they properly categorize input based on the human labels in the training set. Those adjustments minimize a cost function using negative log likelihood. 

Log likelihood estimates the likelihood of a model’s parameters (the weight vectors of softmax, in this case), given output x, the classifications of the training set. It is an inversion of the probability function of output x given certain parameters. 

By maximizing the log likelihood of the parameters with regard to the ground-truth labels x, we train a model that can classify new input better. Taking the logarithm of x, or finding the exponent of 10 that produces x, is a useful way to map very low likelihoods into a narrower and more manageable space; i.e. log(0.0001) can be expressed as -4. It’s a great way to prevent numerical underflow…

We then flip log likelihood to be negative, because most optimization algorithms attempt to minimize a cost or error function, and negative log likelihood plugs in better to a more general framework for optimization, like Deeplearning4j.

Once the weights can no longer be adjusted to reduce error — i.e. once the likelihood of the parameters reaches its peak — the model can be used to categorize input for which no labels exist, inferring the sentiment expressed by text in the wild. 
