---
title: 
layout: default
---

*previous* - [bag of words - tf-idf](../bagofwords-tf-idf.html)
# Word2vec

*to see our Word2vec code, skip to the [training section](../word2vec.html#training1)*

###Introduction to Word2vec

Word2vec is at the heart of text analysis with deep learning. While it does not implement deep learning, Word2vec is crucial to getting input in a numerical form that deep-learning nets can ingest -- the vector. 

Word2vec creates features without human intervention, and some of those features include the context of individual words; that is, it retains context in the form of multiword windows. In machine learning, the meaning of a word is essentially the words that surround it. Given enough data, usage and context, Word2vec can make highly accurate guesses as to a word’s meaning (for the purpose of deep learning, a word's meaning is simply a sign that helps to classify larger entities) based on its past appearances. 

The output of the Word2vec neural net is a vocabulary with a vector attached to it, which can be fed into a deep-learning net for classification/labeling. 

There is also a [skip gram representation](http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf) which is used in the DL4J implementation. This has proven to be more accurate than other models due to the more generalizable contexts generated. 

Broadly speaking, we measure words' proximity to each other through their cosine similarity, which gauges the distance/dissimilarity between two word vectors. A perfect 90-degree angle represents identity; i.e. France equals France, while Spain has a cosine distance of 0.678515 from France, the highest of any other country.

Here's a graph of words associated with "China" using Word2vec:

![Alt text](../img/word2vec.png) 

The other method of preparing text for input to a deep-learning net is called [Bag of Words (BoW)](../bagofwords-tf-idf.html). BoW produces a vocabulary with word counts associated to each element of the text. Its output is a wordcount vector. That said, it does not retain context, and therefore is not useful in a granular analysis of those words' meaning.

## <a name="training1">Training</a> 

Word2Vec trains on raw text. It then records the context, or usage, of each word encoded as word vectors. After training, it's used as lookup table to compose windows of training text for various tasks in natural-language processing.

Assuming a list of sentences, Word2vec is used for lemmatization like this:

<script src="http://gist-it.appspot.com/https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/word2vec/MovingWindowExample.java?slice=45:69"></script>

From there, Word2vec will do automatic multithreaded training based on your sentence data. After that step, you'll want to save Word2vec like this:

       	 SerializationUtils.saveObject(vec, new File("mypath"));

This will save Word2vec to mypath. You can reload it into memory like this:
        
        Word2Vec vec = SerializationUtils.readObject(new File("mypath"));

You can then use Word2vec as a lookup table in the following way:
              
        DoubleMatrix wordVector = vec.getWordVectorMatrix("myword");

        double[] wordVector = vec.getWordVector("myword");

If the word isn't in the vocabulary, Word2vec returns zeros -- nothing more.

### Windows

Word2Vec works with neural networks by facilitating the moving-window model for training on word occurrences. There are two ways to get windows for text:

      List<Window> windows = Windows.windows("some text");

This will select moving windows of five tokens from the text (each member of a window is a token).

You also may want to use your own custom tokenizer like this:

      TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

      List<Window> windows = Windows.windows("text",tokenizerFactory);

This will create a tokenizer for the text, and moving windows based on the tokenizer.

      List<Window> windows = Windows.windows("text",tokenizerFactory);

This will create a tokenizer for the text and create moving windows based on that.

Notably, you can also specify the window size like so:

      TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

      List<Window> windows = Windows.windows("text",tokenizerFactory,windowSize);

Training word sequence models is done through optimization with the [Viterbi algorithm](../doc/org/deeplearning4j/word2vec/viterbi/Viterbi.html).

The general idea is to train moving windows with Word2vec and classify individual windows (with a focus word) with certain labels. This could be done for part-of-speech tagging, semantic-role labeling, named-entity recognition and other tasks.

Viterbi calculates the most likely sequence of events (labels) given a transition matrix (the probability of going from one state to another). Here's an example snippet for setup:

<script src="http://gist-it.appspot.com/https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/word2vec/MovingWindowExample.java?slice=112:121"></script>

From there, each line will be handled something like this:

        <ORGANIZATION> IBM </ORGANIZATION> invented a question-answering robot called <ROBOT>Watson</ROBOT>.

Given a set of text, Windows.windows automatically infers labels from bracketed capitalized text.

If you do this:

        String label = window.getLabel();

on anything containing that window, it will automatically contain that label. This is used in bootstrapping a prior distribution over the set of labels in a training corpus.

The following code saves your Viterbi implementation for later use:
       
        SerializationUtils.saveObject(viterbi, new File("mypath"));

### N-grams & Skip-grams

Words are read into the vector one at a time, *and scanned back and forth within a certain range*, much like n-grams. (An n-gram is a contiguous sequence of n items from a given linguistic sequence; it is the nth version of unigram, bigram, trigram, four-gram or five-gram.)  

This n-gram is then fed into a neural network to learn the significance of a given word vector; i.e. significance is defined as its usefulness as an indicator of certain larger meanings, or labels. 

![enter image description here](http://i.imgur.com/SikQtsk.png)

Word2vec uses different kinds of "windows" to take in words: continuous n-grams and skip-grams. 

Consider the following sentence:

    How’s the weather up there?

This can be broken down into a series of continuous trigrams.

    {“How’s”, “the”, “weather”}
    {“the”, “weather”, “up”}
    {“weather”, “up”, “there”}

It can also be converted into a series of skip-grams.

    {“How’s”, “the”, “up”}
    {“the”, “weather”, “there”}
    {“How’s”, “weather”, “up”}
    {“How’s”, “weather”, “there”}
    ...

A skip-gram, as you can see, is a form of noncontinous n-gram.

In the literature, you will often see references to a "context window." In the example above, the context window is 3. Many windows use a context window of 5. 

### The Dataset

For this example, we'll use a small dataset of articles from the Reuters newswire. 

With DL4J, you can use a **[UimaSentenceIterator](https://uima.apache.org/)** to intelligently load your data. For simplicity's sake, we'll use a **FileSentenceIterator**.

### Loading Your Data

DL4J makes it easy to load a corpus of documents. For this example, we have a folder in the user home directory called "reuters," containing a couple articles.

Consider the following code:

    String reuters= System.getProperty("user.home") +             
    new String("/reuters/");
    File file = new File(reuters);

    SentenceIterator iter = new FileSentenceIterator(new SentencePreProcessor() {
    @Override
    public String preProcess(String sentence) {
        return new 
        InputHomogenization(sentence).transform();
        }
    },file);

In lines 1 and 2, we get a file pointer to the directory ‘reuters’. Then we can pass that to FileSentenceIterator. The SentenceIterator is a critical component to DL4J’s Word2Vec usage. This allows us to scan through your data easily, one sentence at a time.

On lines 4-8, we prepare the data by homogenizing it (e.g. lower-case all words and remove punctuation marks), which makes it easier for processing. 

### Preparing to Create a Word2Vec Object

Next we need the following

        TokenizerFactory t = new UimaTokenizerFactory();

In general, a tokenizer takes raw streams of undifferentiated text and returns discrete, tidy, tangible representations, which we call tokens and are actually words. Instead of seeing something like: 

    the|brown|fox   jumped|over####spider-man.

A tokenizer would give us a list of words, or tokens, that we can recognize as the following list

1. the
2. brown
3. fox
4. jumped
5. over
6. spider-man

A smart tokenizer will recognize that the hyphen in *spider-man* can be part of the name. 

The word “Uima” refers to an Apache project -- Unstructured Information Management applications -- that helps make sense of unstructured data, as a tokenizer does. It is, in fact, a smart tokenizer. 

### Creating a Word2Vec object

Now we can actually write some code to create a Word2Vec object. Consider the following:

    Word2Vec vec = new Word2Vec.Builder().windowSize(5).layerSize(300).iterate(iter).tokenizerFactory(t).build();

Here we can create a word2Vec with a few parameters

    windowSize : Specifies the size of the n-grams. 5 is a good default

    iterate : The SentenceIterator object that we created earlier
    
    tokenizerFactory : Our UimaTokenizerFactory object that we created earlier

This next line is important. 

    vec.setCache(new EhCacheVocabCache());

The EhCacheVocabCache object is important to maintain performance. This object will cache your vocabulary to disk. You need to allocate and create a new cache object for DL4J to work properly.

After this line it's also a good idea to set up any other parameters you need.

Finally, we can actually fit our data to a Word2Vec object

    vec.fit();

That’s it. The fit() method can take a few moments to run, but when it finishes, you are free to start querying a Word2Vec object any way you want. 

    String oil = new String("oil");
    System.out.printf("%f\n", vec.similarity(oil, oil));

In this example, you should get a similarity of 1. Word2Vec uses cosine similarity, and a cosine similarity of two identical vectors will always be 1. 

Here are some functions you can call:

1. *similarity(String, String)* - Find the cosine similarity between words
2. *analogyWords(String A, String B, String x)* - A is to B as x is to ?
3. *wordsNearest(String A, int n)* - Find the n-nearest words to A

### Troubleshooting & Tuning Word2Vec

*Q: I get a lot of stack traces such as*

    java.lang.StackOverflowError: null
    at java.lang.ref.Reference.<init>(Reference.java:254) ~[na:1.8.0_11]
    at java.lang.ref.WeakReference.<init>(WeakReference.java:69) ~[na:1.8.0_11]
    at java.io.ObjectStreamClass$WeakClassKey.<init>(ObjectStreamClass.java:2306) [na:1.8.0_11]
    at java.io.ObjectStreamClass.lookup(ObjectStreamClass.java:322) ~[na:1.8.0_11]
    at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1134) ~[na:1.8.0_11]
    at java.io.ObjectOutputStream.defaultWriteFields(ObjectOutputStream.java:1548) ~[na:1.8.0_11]

*A:* Look inside the directory where you started your Word2vec application. This can, for example, be an IntelliJ project home directory or the directory where you typed Java at the command line. It should have some directories that look like:

       ehcache_auto_created2810726831714447871diskstore  
       ehcache_auto_created4727787669919058795diskstore
       ehcache_auto_created3883187579728988119diskstore  
       ehcache_auto_created9101229611634051478diskstore

You can shut down your Word2vec application and try to delete them.

*Q: Not all of the words from my raw text data are appearing in my Word2vec object…*

*A: Try to raise the layer size via **.layerSize()** on your Word2Vec object like so*

        Word2Vec vec = new Word2Vec.Builder().layerSize(300).windowSize(5).layerSize(300).iterate(iter).tokenizerFactory(t).build();

###Fine-tuning DBNs

Now that you have a basic idea of how to set up Word2Vec, here's one example of how it can be used to finetune a deep-belief network:

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/word2vec/MovingWindowSingleThreaded.java?slice=96:110"></script>

There are three parameters to pay special attention to here. The first is the number of words to be vectorized in the window, which you enter after getWindow. The second is the number of nodes contained in the layer, which you'll enter after getLayerSize. Those two numbers will be multiplied to obtain the number of inputs. Finally, remember to make your activation algorithm *hardtanh*. 

Word2Vec is especially useful in preparing text-based data for information retrieval and QA systems, which DL4J implements with [deep autoencoders](../deepautoencoder.html). For sentence parsing and other NLP tasks, we also have an implementation of [recursive neural tensor networks](../recursiveneuraltensornetwork.html).
