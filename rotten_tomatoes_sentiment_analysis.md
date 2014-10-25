---
layout: default
---

<!-- DataTables CSS -->
<link rel="stylesheet" type="text/css" href="//cdn.datatables.net/1.10.3/css/jquery.dataTables.css">
  
<!-- jQuery -->
<script type="text/javascript" charset="utf8" src="//code.jquery.com/jquery-1.10.2.min.js"></script>
  
<!-- DataTables -->
<script type="text/javascript" charset="utf8" src="//cdn.datatables.net/1.10.3/js/jquery.dataTables.js"></script>

# Movie Review Sentiment Analysis 

In this blog post, we're going to walk through a sentiment analysis of movie reviews using the Rotten Tomatoes dataset. 

You'll need to download the dataset from Kaggle (registration required):

          https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

The dataset is split into a training set and a test set already, which makes our lives easier.  Let's download the data and load it into our nets. 

Go ahead and unzip it:

          unzip train.tsv.zip

In our folder we see a train.tsv file. What does the data look like? A one-word command will show us.

          head train.tsv

The command 'head' should output the following table:

<table id="first_table" class="display">
    <thead>
        <tr>
            <th>PhraseId</th>
            <th>SentenceId</th>
            <th>Phrase</th>
            <th>Sentiment</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>1</td>
            <td>A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story .</td>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
            <td>1</td>
            <td>A series of escapades demonstrating the adage that what is good for the goose</td>
            <td>2</td>
        </tr>
        <tr>
            <td>3</td>
            <td>1</td>
            <td>A series</td>
            <td>2</td>
        </tr>
        <tr>
            <td>4</td>
            <td>1</td>
            <td>A</td>
            <td>2</td>
        </tr>
        <tr>
            <td>5</td>
            <td>1</td>
            <td>series</td>
            <td>2</td>
        </tr>
        <tr>
            <td>6</td>
            <td>1</td>
            <td>of escapades demonstrating the adage that what is good for the goose</td>
            <td>2</td>
        </tr>
        <tr>
            <td>7</td>
            <td>1</td>
            <td>of</td>
            <td>2</td>
        </tr>
        <tr>
            <td>8</td>
            <td>1</td>
            <td>escapades demonstrating the adage that what is good for the goose</td>
            <td>2</td>
        </tr>
        <tr>
            <td>9</td>
            <td>1</td>
            <td>escapades</td>
            <td>2</td>
        </tr>
    </tbody>
</table>

Let's walk through this. We have a SentenceID of 1 in every row, which means we are dealing with the same sentence throughout. The entire sentence is presented as a phrase in Row 2, Column 3. Each subsequent phrase shown is Column 3 is a subset of that original sentence. 

Our table is a partial preview of the sentence's subsets, and it stops short of presenting the phrases that constitute the second half of the sentence. In addition, this is a supervised dataset, so each phrase has been assigned a sentiment label by a real, live human being. 

The sentiment labels are:

| Label    |   Sentiment   |
|----------|:-------------:|
|  0 |  negative |
|  1 |    somewhat negative   |
| 2 | neutral |
| 3 |    somewhat positive   |
| 4 | positive |


This happens to be quite nuanced: many sentiment analysis problems are binary classifications; i.e. 1 or 0, positive or negative.

In our preview of Sentence 1, the sentence itself has been assigned the label of "somewhat negative," which is appropriate, given the second half of the sentence is critical of the film in question. The first half is not critical, however, and its subphrases have all been labeled "neutral," or 2. 

From Kaggle:

         The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.


     train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.


test.tsv contains just phrases. You must assign a sentiment label to each phrase.

Let's introduce the methods we will be using in this blog post.

Dataset - How to initially tackle the problem
================================

We notice 2 columns describing the sentence. The phrase and the sentence id. A phrase id is a sub window we use to break up the text such that each sub window of a 
sentence is an example of a "context". Each of these sub contexts has a possible label. We are going to want to build tools to break these in to contexts mapping by sentence id.

For dbns, we will be building a simple moving window and labeling each window where the span has a particular label.

For Recursive Neural Tensor Networks, I will demonstrate how to label trees as sub contexts. Each sub node of the tree matches a span which 

also has a particular label.

The reason we focus on sub contexts for sentiment analysis comes down to the fact that context and subcontexts are what capture sentiment.

The simple is a negation of a positive word. We want to be able to capture this in our feature space. This is why we will focus on word vectors

for our sentiment analysis.

Luckily, the vectorization code is already written for this and we can just focus on the high level logic of word vectors matched to sub contexts and 

benchmarking the models.







Bag of Words
===============================

This is a pretty traditional baseline. A bag of words is a representation used in a lot of natural language processing tasks. This is meant to be used at the "document" level.

A document in this case is an individual unit of text that does not retain context and is typically the tfidf or the count of the words where present in the vector. The

number of columns in the feature vector is relative to the vocab.

This is a very sparse feature set (lots of zeros)




Moving Window DBN
=========================================

We will be using a sequence moving window approach with the viterbi algorithm to identify classes. This is similar to the approach used by collobert et. al in nlp almost from scratch.

The features we will be using with this are word vectors (explained later)


Recursive Neural Tensor Networks
===========================================


Done by socher et. al, this is a tree parsing algorithm where the neural nets are attached individual nodes of a binary tree. The leaves are word vectors.





Let's do a deeper dive in to each of these approaches now. First we will establish some common ground terms that we will use for the rest of the tutorial.

First, we need to motivate a few concepts.

NLP Pipelines
=============================================



NLP Pipelines are how you pre process text in to a known format that you can use for classification. We will be doing 2 things, first breaking things up in to documents, and then following up by computing a vocab.

This will be used for bag of words as well as word vectors.



Vocab Computation
================================================

This is composed of the unique set of words in a corpus (a textual dataset) With deeplearning4j, we supply tools for computation of many of these things. We will be computing each of these differently.

Breaking out the corpus
===================================

As of right now, each of our phrases is still in a csv. Let's work on parsing out the text in to a more raw form. We will do this by implementing stepping through the creation of a dataset iterator that will give us a reproducible data pipeline that can be tested.



DataSetIterator
===========================================


When you are creating a machine learning model, you need to vectorize the data in some way. Vectorization is the process of breaking up unstructured data in to a set of features. The features are a vector (sound familiar?) The 2 representations we will be focusing on here are word vectors(context, vector per word) and word count vectors (document level, no context).


One component I like breaking out is the idea of the data retrieval. When we iterate over  a dataset, there is an order from which we will be returning things. Data can live in multiple locations though. If this is the case, we want the data pipeline retrieval for each potential end point to be isolated and testable. This is represented in something called a datafetcher.

DataFetcher
==========================================


A Data fetcher handles data retrieval. Data may live in any number of places including AWS, your file system, or even just mysql. When you do feature vector composition from different sources,
you want to ensure these aspects of a data pipeline are isolated. With this in mind, let's look at how to fetch data from a csv and parse it.




We will be building a data fetcher to do this.



CSV Processing
==========================================

As part of deeplearning4j for handling of csv values we have a wonderful csv library that allows us to do the following:


        CSV csv1 = CSV.separator('\t')
                .ignoreLeadingWhiteSpace().skipLines(1)
                .create();
        csv1.read(csv,new CSVReadProc() {
            @Override
            public void procRow(int rowIndex, String... values) {
                
            }
        });

csv here is a file object.



The callback then will allow us ot get access to the data. Our goal then will be to collect the text and create what amounts to a corpus.

In our call back, we will want to grab the text and treat each line as a document.

Due to the nature of the beast, we will just create a list of the documents since there's not that much in memory. 

According to kaggle, competition we will be classifying phrases. Therefore, this is what we will be saving in to the list at each individal row.


That leads us to a class with an abstract data type called a text retriever that contains the following:

Map<String,Pair<String,String>> mapping the phrase id, to the content and the associated label.



The body of our csv parser in procRow earlier will then be:

            pair.put(values[0],new Pair<>(values[2],values[3]));


This is our map of phrases to text and label.

We can use this information generally without coupling this to any particular implementation of our classifier. Let's now test this.

 


Testing
=============================================


Being consistent with our separation of concerns and testing, let's build a unit test for this.


Since our dataset is on our class path, we can use a few neat tricks to retrieve it. Deeplearning4j leverages spring for a few reflection utilities as well as 

for more robust of classpath discovery of components. With that in mind our test will looking like the following:






    @Test
    public void testRetrieval() throws  Exception {
         Map<String,Pair<String,String>> data = retriever.data();
        Pair<String,String> phrase2 = data.get("2");
        assertEquals("A series of escapades demonstrating the adage that what is good for the goose",phrase2.getFirst());
        assertEquals("2",phrase2.getSecond());

    }




One thing we might want to do though (later on) is to get things like just phrases or just labels. Let's make this something easy to do for ourselves later. Our goal is to build a set of abstractions that we can think in terms of components and not individual steps.


For brevity, I will just show the associated test and let it speak for itself:


   @Test
    public void testNumPhrasesAndLabels() {
        assertEquals(NUM_TRAINING_EXAMPLES,retriever.phrases().size());
        assertEquals(NUM_TRAINING_EXAMPLES,retriever.labels().size());
    }


In many problems, we will often want to test multiple methods of classifiation to know what one works well (or even use all of them for an ensemble which will always be better anyways)



Sentence Iterator
===================================

Now that we have our basic csv processing, let's start to turn this in to something that vaguely resembles a data pipeline. One thing that gives us a semblance of a corpus is this idea of a sentence iterator. In our case a sentence is a document. So what does this look like? 

We will be implementing a label aware sentence iterator. This gives us a concept of supervised learning where a document has a label. The core concept of a sentence iterator, is it know where it is in the corpus and will always be able to give us the next sentence and also tell us when its done. This is a lot of responsibility, we will want to be able to isolate this.

The core bits of logic are here:
 
    private List<String> phrases;
    private List<String> labels;
    private int currRecord;
    private SentencePreProcessor preProcessor;



Our goal is to keep track of which sentence we are on and the current label. We do this with the curr record position. We use the text retriever we built earlier for retrieval of data.

This separates the responsibility of data retrieval and iteration which encourages good practice in software engineering.



The juicy parts:


 try {
            TextRetriever retriever = new TextRetriever();
            this.phrases = retriever.phrases();
            this.labels = retriever.labels();
        } catch (IOException e) {
            e.printStackTrace();
        }


Whoa that was easy. That gives us exactly what we need to iterate over. What's nice is we've wrapped this in a nicer interface that is standard across any data pipeline you will build with
deeplearning4j.

The test:


    @Test
    public void testIter() {
        LabelAwareSentenceIterator iter = new RottenTomatoesLabelAwareSentenceIterator();
        assertTrue(iter.hasNext());
        //due to the nature of a hashmap internally, may not be the same everytime
        String sentence = iter.nextSentence();
        assertTrue(!sentence.isEmpty());
    }






Now we know one of our building blocks works. Now we can worry about higher level concepts like word vectors and bag of words.



Let's build a bag of words datafetcher first. This will be easier than we think.

public class RottenTomatoesBagOfWordsDataFetcher extends BaseDataFetcher {

    private LabelAwareSentenceIterator iter;
    private BagOfWordsVectorizer countVectorizer;
    private TokenizerFactory factory = new DefaultTokenizerFactory();
    private DataSet data;

    public RottenTomatoesBagOfWordsDataFetcher() {
        iter = new RottenTomatoesLabelAwareSentenceIterator();
        countVectorizer = new BagOfWordsVectorizer(iter,factory, Arrays.asList("0", "1", "2", "3", "4"));
        data = countVectorizer.vectorize();

    }


    @Override
    public void fetch(int numExamples) {
        //set the current dataset
        curr = data.get(ArrayUtil.range(cursor,numExamples));

    }
}



As we can see, only a few lines. You'll notice a few components, let's break this down a bit.
One is the iterator, we built this earlier to track where we are currently when iterating over the data, this also associates a string with a dataseet.

Next is the count vectorizer, this is the workhorse, let's load the data in to memory with vectorize and iterate as necessary.


Note that this WILL be ram intensive and I wouldn't recommend running this part unless you're on a fairly beefy server. I will run these benchmarks for you here.


I would recommend pruning words from your vocab via tfidf to get a good approximation of your data, here I will use the whole dataset for simplicity though.

Unfortunately, this is due to a limitation in nd4j only supporting dense matrices, we will be working on sparse formats at a later date.

As nd4j is a blas focused framework initially, that is what we will be supporting for now. With that in mind, let's move forward.

So what exactly did we do here? 

We wrote something that could parse csvs, take the text, map it to a label, and then iterate through it producing a matrix.


One key component we built is a vocab. This vocab has around 17k words in it. For bag of words matrices, this will be a sparse representation of 150k x 17k.

Not a lot of bang for our buck here. We'lll have to see how the classifier (DBN) does.




Word Vectors
=============================================================

Let's play around with word vectors now. Remember, word vectors are used for featurization of textual contexts. We will end up using the viterbi algorithm with voting on moving window

for document classification here.



Firstly, since this is a word vector based approach we are going to be using word2vec. We are going to want to dig in to how well word2vec trains.

Unlike bag of words where features are deterministic, word vectors are a form of neural net which means training coefficients.



One thing that will help is to visualize everything. Let's visualize the 16000 word vocab with d3. This will also involve an algorithm called tsne

to see the proximity of words to other words. We need to ensure that the words themselves are coherent.




TK: Renders

Word vectors are used in sequential applications of text. They can be used in document classification with a proper ensemble (voting) method as well
by optimizing for a maximum likelihood estimator over the windows and labels.

So what does word2vec look like in code?

The key snippet is here:

 Word2vec vec = new Word2Vec.Builder().iterate(iter).tokenizerFactory(factory)
                .learningRate(1e-3).vocabCache(new InMemoryLookupCache(300))
                .layerSize(300).windowSize(5).build();
 vec.fit();


Explaining this a bit, you'll notice we specify a document iterator, a tokenizer factory, a learning rate, among other things.

I will go over each of these parameters now:

iter: DocumentIterator this is our raw textual pipeline
factory: our tokenizer factory, handles tokenizing text
learning rate: step size
cache: this is where all of our metadata about vocabulary is stored including word vectors, tfidf scores, document frequencies as well as where doucments occurred.
layer size: this is the number of features per word
window size: the window size for iterating over text, this is how long of contexts to train on.

Remember wordvec represents word usage.







Moving Window DBN
=========================================================

SO what is a moving window and how do we do it? A moving window is a sliding window over text such that we take a portion of the text of a certain size and classify that as one example.

Think of this very similar to the concept of ngrams.

An example moving window of 3:

Sentence:

The cat sat on the mat.

<s> The cat
The cat sat
sat on the
on the mat
the mat .
mat . <s>

Notice I append and prepend padding to the text. I do this in deeplearning4j as well.

In code this would look like:

List<Window> windows = Windows.windows("The cat sat on the mat.",3);

Note the second parameter, this is the window size.

In our case, each window becomes an example. What we want to do is do a look up for each token in the window and use that to create an example feature vector. 

This window would then have a label.

Afterwards, we will have a sequence of labels. From there, we will use viterbi which optimizes sequence prediction of labels treating them aas individual events.


This is actually pretty easy to do relative to the work we've already done.



 while(iter.hasNext()) {
            DataSet next = iter.next();
            d.fit(next);
               Pair<Double,INDArray> labels =  v.decode(next.getLabels());

}


Just train the likelihoods of the labels along with the data sets and you will get one classification for each phrase as follows.
Use labels.getFirst() to get the overall vote and most likely sequence.


Recursive Neural Tensor Networks
========================================================================


Another way of doing sequential text classification is recursive neural tensor networks. This requires our 
sentence be parsed in to a tree. The leaves are individual words with higher level nodes comprising contexts.

Of course this also means our vectorization strategy is slightly different. Given below:
 TreeVectorizer vectorizer = new TreeVectorizer(new TreeParser());

        while(iter.hasNext()) {
            List<Tree> trees = vectorizer.getTreesWithLabels(iter.nextSentence(),iter.currentLabel(), Arrays.asList("0","1","2","3","4"));
            t.fit(trees);
 }


Going line by line, all we are doing here is creating a tree vectorizer which knows hw to handle word vectors and tree parsing.

We require a sentence iterator for iterating over the text instead of the windows. The vectors and probabilities 

are dervied from the neural network itself.

Notice here that we pass in the sentence, the current label (note that the sentence is "label aware"0 and the list of possible lables. 

This is for creating the outcome matrices.



Evaluation and Classification
==========================================================================


Next we will be getting in to the classification and evaluation. What we've explained up to this point is the vectorization strategies involved with each method. 

For classification and the like, each neural net natrually is going to classify relative to the examples. In a dbn, this is the windows, and in RNTNs, this is going to be

individual nodes of the tree represented by the context.

Our overall goal, is to classify phrases, we do this by aggregating the contexts and essentially voting on the results based on the likelihoods of different contexts

having a particular label.



I will be keeping each of these examples side by side so we can evaluate them on the same terms (F1 scores) while talking about their minute differences

in classification and testing. This is a great way to keep the overall goal in mind while not getting lost in the minute details of each model.

Our evaluation metric is going to be the F1 score. This is typically used in multinomial classification problems.


This will tell us (relative to precision and recall) how well our classifier did based on a true set of labels and a held out test set.


First we are going to need to load the test set. Luckily, we can reuse all of the work we did for training. Data pipelines should always be 

reproducible for both training and testing (mainly for consistency)





For our first 2 methods, we will be using the evaluation class which handles evaluation of our outcome matrices.

For the RNTN, we will need to use a slightly different technique that scores each indivdial node of a tree.

This is consistent with our examples where we needed to vectorize the recursive net differently than the feed forward examples (both bag of words AND moving window)


















