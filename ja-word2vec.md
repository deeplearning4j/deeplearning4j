---
title:
layout: ja-default
---

# Word2Vec

コンテンツ

* <a href="#intro">Word2Vec入門</a>
* <a href="#anatomy">Anatomy of Word2vec</a>
* <a href="#code">Training</a>
* <a href="#windows">Moving Windows</a>
* <a href="#grams">N-grams & Skip-grams</a>
* <a href="#load">Loading Your Data</a>
* <a href="#trouble">Troubleshooting & Tuning Word2Vec</a>
* <a href="#dbn">Fine-tuning DBNs (with code!)</a>
* <a href="#next">Next Steps</a>

###<a name="intro">Word2Vec入門</a>

[Deeplearning4j](http://deeplearning4j.org/quickstart.html)では、分散表現である Word2vec を Java で実装しています。これは GPU 上でも動作します。 Word2vec は Tomas Mikolov 氏が率いる Google の研究チームによってに最初に生み出されました。

Word2vec は テキストを処理するニューラルネットで、それらのテキストが深層学習(deep-learning)アルゴリズムで処理される前に利用されます。Word2vec 自体は深層学習を実装していませんが、テキストを深層学習のニューラルネットが理解出来るような数値化された表現(つまりベクトル表現)に変換します。

Word2vec は人間の介在なしに、個々の単語のコンテキストを含む特徴(features)を生成します。コンテキストは、複数単語からなるウィンドウの形をしています。Word2vec は、十分なデータ、利用例、コンテキストが与えられれば、かなり高い精度で単語の意味を、過去の出現例を基に推測することが出来ます。ここでの意味とは、深層学習で利用するための意味のことで、単に大きなエンティティを分類をするのに役立つようなシグナルのことです(例えば、文書をクラスタに分類するような)。

Word2vec は文章の文字列を入力として受け取ります。各文(単語列)はn次元ベクトル化され、他のベクトル化された単語列とn次元ベクトル空間で比較されます。関連する単語や単語列は、その空間上で近い場所に現れます。ベクトル化することで、単語列同士の類似度をある程度の精度で測定することを可能にし、クラスタ化するのです。それらのクラスタは検索、感情分析、レコメンデーションを行う際のベースになります。

Word2vec ニューラルネットの出力はベクトルで表現された語彙で、それは深層学習のニューラルネットに分類/ラベル付するために入力することが出来ます。

skip-gram 表現は Mikolov 氏によって普及され、DL4Jの実装でも利用されています。これは、他の表現モデルよりも、より一般化可能なコンテキストが生成される為、より正確だと証明されています。

概して言えば、私達は単語同士の近さをコサイン類似度によって評価しています。コサイン類似度は2つの単語ベクトル間の距離(相違度)を測定します。完全な90度を成すベクトル同士は同一であることを表します。すなわち"France"は"France"と等しく、"France"からみて"Spain"は0.678515のコサイン距離の位置にありまる(他の国とくらべて一番遠い)。

これが Word2vec を使って "China" と関連付けられた単語のグラフです。

![Alt text](../img/word2vec.png)

###<a name="anatomy">Anatomy of Word2vec</a>

What do we talk about when we talk about Word2vec? Deeplearning4j's natural-language processing components:

* **SentenceIterator/DocumentIterator**: Used to iterate over a dataset. A SentenceIterator returns strings and a DocumentIterator works with inputstreams. Use the SentenceIterator wherever possible.
* **Tokenizer/TokenizerFactory**: Used in tokenizing the text. In NLP terms, a sentence is represented as a series of tokens. A TokenizerFactory creates an instance of a tokenizer for a "sentence."
* **VocabCache**: Used for tracking metadata including word counts, document occurrences, the set of tokens (not vocab in this case, but rather tokens that have occurred), vocab (the features included in both bag of words as well as the word vector lookup table)
* **Inverted Index**: Stores metadata about where words occurred. Can be used for understanding the dataset. A Lucene index with the Lucene implementation[1] is automatically created.

Briefly, a two-layer neural net is trained with <a href="../glossary.html#downpoursgd">Gradient Descent</a>. The connection weights for the neural net are of a specified size. <em>syn0</em> in Word2vec terms is the wordvector lookup table, <em>syn1</em> is the activation, and a hierarchical Softmax trains on the two-layer net to calculate the likelihoods of various words being near one another. The Word2vec implementation here uses <a href="../glossary.html#skipgram">skipgrams</a>.

## <a name="code">Training</a>

Word2Vec trains on raw text. It then records the context, or usage, of each word encoded as word vectors. After training, it's used as lookup table to compose windows of training text for various tasks in natural-language processing.

After lemmatization, Word2vec will conduct automatic multithreaded training based on your sentence data. Then you'll want to save the model. There are a few different components to Word2vec. One of these is the vocab cache. The normal way to save models in deeplearning4j is via the SerializationUtils (java serialization akin to python pickling)

        SerializationUtils.saveObject(vec, new File("mypath"));

This will save Word2vec to mypath. You can reload it into memory like this:

        Word2Vec vec = SerializationUtils.readObject(new File("mypath"));

You can then use Word2vec as a lookup table in the following way:

        INDArray wordVector = vec.getWordVectorMatrix("myword");
        double[] wordVector = vec.getWordVector("myword");

If the word isn't in the vocabulary, Word2vec returns zeros -- nothing more.

###<a name="windows">Windows</a>

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

Training word sequence models is done through optimization with the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm).

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

### <a name="grams">N-grams & Skip-grams</a>

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

A skip-gram, as you can see, is a form of discontinous n-gram.

In the literature, you will often see references to a "context window." In the example above, the context window is 3. Many windows use a context window of 5.

### <a name="dataset">The Dataset</a>

For this example, we'll use a small dataset of articles from the Reuters newswire.

With DL4J, you can use a **[UimaSentenceIterator](https://uima.apache.org/)** to intelligently load your data. For simplicity's sake, we'll use a **FileSentenceIterator**.

### <a name="load">Loading Your Data</a>

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

### <a name="prepare">Preparing to Create a Word2Vec Object</a>

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

### <a name="create">Creating a Word2Vec object</a>

Now we can actually write some code to create a Word2Vec object. Consider the following:

    Word2Vec vec = new Word2Vec.Builder().windowSize(5).layerSize(300).iterate(iter).tokenizerFactory(t).build();

Here we can create a word2Vec with a few parameters

    windowSize : Specifies the size of the n-grams. 5 is a good default

    iterate : The SentenceIterator object that we created earlier

    tokenizerFactory : Our UimaTokenizerFactory object that we created earlier

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

### <a name="trouble">Troubleshooting & Tuning Word2Vec</a>

*Q: I get a lot of stack traces like this*

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

*A:* Try to raise the layer size via **.layerSize()** on your Word2Vec object like so

        Word2Vec vec = new Word2Vec.Builder().layerSize(300).windowSize(5)
                .layerSize(300).iterate(iter).tokenizerFactory(t).build();

### <a name="dbn">Fine-tuning DBNs</a>

Now that you have a basic idea of how to set up Word2Vec, here's one example of how it can be used with DL4J's API:

<script src="http://gist-it.appspot.com/https://github.com/SkymindIO/dl4j-examples/blob/master/src/main/java/org/deeplearning4j/word2vec/Word2VecExample.java?slice=21:90"></script>

There are a couple parameters to pay special attention to here. The first is the number of words to be vectorized in the window, which you enter after WindowSize. The second is the number of nodes contained in the layer, which you'll enter after LayerSize. Those two numbers will be multiplied to obtain the number of inputs.

Word2Vec is especially useful in preparing text-based data for information retrieval and QA systems, which DL4J implements with [deep autoencoders](../deepautoencoder.html). For sentence parsing and other NLP tasks, we also have an implementation of [recursive neural tensor networks](../recursiveneuraltensornetwork.html).

### <a name="next">Next Steps</a>

An example of sentiment analysis using [Word2Vec is here](http://deeplearning4j.org/sentiment_analysis_word2vec.html).

(We are still testing our recent implementations of Doc2vec and GLoVE -- watch this space!)
