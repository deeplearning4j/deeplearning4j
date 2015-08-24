---
layout: default
---

# Sentiment Analysis With Word2vec

*This page is a work in progess.*

Contents

* <a href="#vectors">Vectors</a>
* <a href="#trig">Trigonometric Functions</a>
* <a href="#bow">Bag of Words v. Word2vec</a>
* <a href="#count">Beyond Word Counts</a>
* <a href="#sentiment">Sentiment Analysis</a>
* <a href="#softmax">Softmax Logistic Regression</a>
* <a href="#code">Just Give Me The Code</a>
* <a href="#resource">Other Resources</a>

This tutorial covers sentiment analysis with Word2vec and logistic regression. It is written for programmers, but assumes knowledge of only basic mathematical concepts. Its purpose is to demonstrate how Word2vec can be used for opinion mining on text in the wild. 

Sentiment has obvious applications for market research, business intelligence, product development, reputation management, political campaigns and sociological studies. 

## <a name="vectors">Vectors</a>

[Word2vec](../word2vec.html) learns to represent words as vectors.

A vector is a data structure with at least two components, as opposed to a *scalar*, which has just one. For example, a vector can represent velocity, an idea that combines speed and direction: *wind velocity* = (50mph, 35 degrees North East). A scalar, on the other hand, can represent something with one value like temperature or height: 50 degrees Celsius, 180 centimeters.

Therefore, we can represent two-dimensional vectors as arrows on an x-y graph, with the coordinates x and y each representing one of the vector's values. 

![Alt text](../img/vector.jpeg)

These vectors relate mathematically, and similarities between them (and therefore between anything you can vectorize, including words) can be measured with precision. 

![Alt text](../img/two_vectors2.png)

As you can see, these vectors differ from one another in both their length, or magnitude, and in their angle, or direction. The angle is what concerns us here. Differences between word vectors, as they swing around the origin like the arms of a clock, can be thought of as differences in degrees. 

Like ancient navigators gauging the stars by a sextant, we will measure the angular distance between words using something called *cosine similarity*.

![Alt text](../img/angular_distance.png)

To find that distance knowing only the word vectors, we need the equation for vector dot multiplication (multiplying two vectors to produce a single, scalar value).

![Alt text](../img/colored_dot_product.png)

In Java, you can think of the formula to measure cosine similarity like this:

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

## <a name="trig">Trigonometric Functions</a>

Let's do a quick trig review. 

Trigonometric functions like *sine*, *cosine* and *tangent* are ratios that use the lengths of a side of a right triangle (opposite, adjacent and hypotenuse) to compute the shape’s angles. By feeding the sides into ratios like 

![Alt text](../img/trig_functions2.png)

we can also know the angles at which those sides intersect. Remember [SOH-CAH-TOA](http://mathworld.wolfram.com/SOHCAHTOA.html)?

Cosine is the angle attached to the origin, which makes it useful here. (We normalize the measurements so they come out as percentages, where 1 means that two vectors are equal, and 0 means they are perpendicular, bearing no relation to each other.)

## <a name="bow">Bag of Words vs. Word2vec</a>

For this tutorial, we are interested in sentiment analysis and classification on the document level. 

Typically, people rely on word count to create vectors representing documents, and measure their similarities and differences by the frequency with which terms appear. To illustrate this word count approach with a toy example, let’s say we care about two words, Moscow and Beijing. 

First, we count the number of times those words appear in two documents. Let’s say the x axis represents Moscow and the y axis Beijing. If Moscow appears once in document one and five times in document two, while Beijing appears three times in the first and twice in the second, then we have our two vectors: doc1 = (1,3) and doc2 = (5,2).

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

Creating a document-vector representation for sentiment analysis is as simple as adding together the feature vectors for each word in the document and dividing that vector by the number of word vectors extracted from the document. 

EQUATION HERE // Jeff?

This document vector (not to be confused with doc2vec) can then be compared to vectors representing documents in, say, a labeled set.

## <a name="sentiment">Sentiment Analysis </a>

For neural networks to learn sentiment, you need a labeled dataset to conduct supervised learning; i.e. you must have a set of documents or words that humans have associated with emotional signals, be they as simple as *positive* and *negative*, or as nuanced as frustration, anger, delight, satisfaction and lacadaisical whimsicality.

So the first step is to pick the categories you care about. Then you create a dataset in which the examples have been tagged with those labels. (Mechanical Turk is useful here...) If you don't mind using someone else's categories, you can download TKTKTK.

Once you've pulled together a labeled corpus, then you feed the words into Word2vec to generate feature vectors. Those feature vectors are added and the aggregate word vector is divided by the number of words. That "document vector"  serves as the input for logistic regression. 

CODE HERE

## <a name="softmax">Softmax Logistic Regression</a>

[Logistic regression](http://gormanalysis.com/logistic-regression-fundamentals/), despite its misleading name, classifies things. 

![Alt text](../img/logistic_regression2.png)

Given one or more input variables, it estimates the probability that the input belongs to one category or another. Each variable of the input is a component of a vector, one of the feature vectors mentioned above. Imagine a feature vector of Betas, the weights that modify the input.

![Alt text](../img/logistic_regression3.png)

The simplest form of logistic regression is binary, predicting categories such as *spam* or *not_spam*. 

A more complex form that buckets input into more than two categories is called multinomial logistic regression, or *softmax*. That’s what we’ll be using here. 

Softmax dot multiplies an input vector by a weight vector, using separate weight vectors for each respective output. It then squashes the results into a narrow range, and assumes the category whose weight vector produces the highest result is the correct classification. Thus the *max*.

With supervised learning, we adjust the weight vectors of softmax until they properly categorize input based on the human-labelled training set. Those adjustments minimize a cost function using something called *negative log likelihood*. 

Log likelihood estimates the likelihood of a model’s parameters (the weight vectors of softmax, in this case), given output x, the classifications of the training set. It is an inversion of the probability function of output x given certain parameters. 

By maximizing the log likelihood of the parameters with regard to the ground-truth labels x, we train a model that can classify new input better. 

Taking the logarithm of x, or finding the exponent of 10 that produces x, is a useful way to map very low likelihoods into a narrower and more manageable space; i.e. log(0.0001) can be expressed as -4. It’s a great way to prevent [arithmetic underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow), which is a risk when you deal with very low probabilities.

We then flip log likelihood to be negative, because most optimization algorithms attempt to minimize a cost or error function, and negative log likelihood plays nice with a more general framework for optimization, like Deeplearning4j.

Once the weights can no longer be adjusted to reduce error — i.e. once the likelihood of the parameters given the output reaches its peak — the model can be used to categorize input for which no labels exist, inferring sentiment expressed by text in the wild. 

## <a name="code">Just Give Me the Code</a>

## <a name="resource">Other Resources</a>

* 
