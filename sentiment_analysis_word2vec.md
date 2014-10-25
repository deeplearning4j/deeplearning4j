---
layout: default
---

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

Let's walk through this. 

Two columns describe the sentence: PhraseID and SentenceID. A PhraseID is a sub-window that takes just one piece of a larger passage, such that each sub-window of a sentence is an example of a "context." 

In the table above, we have a SentenceID of 1 in every row, which means we are dealing with the same sentence throughout. The entire sentence is presented as a phrase in Row 2, Column 3. Each subsequent phrase shown is Column 3 is a subset of that original sentence: the sub-windows. 

Our table is only a partial preview of the sentence's subsets, as it happens to end before presenting the phrases that constitute the second half of the sentence. 

This is a supervised dataset, each sub-window has been assigned a label denoting its sentiment by a real, live human being. Here's a table mapping sentiment to numeric labels:

| Label |  Sentiment |
|:----------:|:-------------:|
|  0 |  negative |
|  1 |    somewhat negative   |
| 2 | neutral |
| 3 |    somewhat positive   |
| 4 | positive |

This label system is fairly nuanced: many sentiment analysis problems are binary classifications; that is, 1 or 0, positive or negative, with no finer gradations.

In our preview of Sentence 1, the sentence itself has been assigned the label of "somewhat negative," which is appropriate, given the second half of the sentence is critical of the film in question. The first half is not critical, however, and its subphrases have all been labeled "neutral," or 2. 

From Kaggle:

*The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.*

* *train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.*

* *test.tsv contains just phrases. You must assign a sentiment label to each phrase.*

Now we'll introduce the methods used here.

## Dataset: How to Approach the Problem

We want to build tools to break the sentences in the reviews into sub-windows or contexts, grouping them by their SentenceID.

* For deep-belief networks (DBNs), we'll build a simple moving window and label each window within a sentence by sentiment. We'll be using a sequence moving window approach with the Viterbi algorithm to label phrases. This is similar to the approach used by Ronan Collobert et al in the paper [Natural Language Processing (Almost) From Scratch](https://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/35671.pdf). The features are word vectors, explained below.
* For recursive neural tensor networks (RNTNs), we'll demonstrate how to label trees as sub-contexts. Each sub-node of the tree matches a span that also has a particular label. [Created by Richard Socher et al](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf), a recursive neural tensor network is a tree-parsing algorithm in which neural nets are attached to the individual nodes on a binary tree. The leaves of the tree are word vectors. 

We focus on sub-contexts for sentiment analysis, because context and subcontexts, not isolated words, are what capture sentiment signals best. A context including "not" or "no" will nullify a positive word, requiring a negative label. We need to be able to capture this sentiment in our feature space. 

The vectorization code to capture context is already written, allowing us to concentrate on the high-level logic of word vectors matched to sub-contexts, and then on benchmarking the accuracy of our models.

Before we do a deeper dive into these approaches, we'll have to define and explain some terms.

### Bag of Words (BoW)

Bag of Words is the baseline representation used in a lot of natural-language processing tasks. It's useful at the "document" level, but not at the level of sentences and their subsets.

With Bag of Words, a document is the atomic unit of text. BoW doesn't dig deeper. It retains no context for individual words or phrases within the document. Bag of Words is essentially a word count contained in a vector. 

A slightly more sophisticated version of BoW is "term frequency–inverse document frequency," or TF-IDF, which lends weight to a single term's frequency within a given document, while discounting terms that are common to all documents (a, the, and, etc.). The number of columns in the feature vector will vary with the size of the vocabulary. This produces a very sparse feature set, with a lot of 0s for the words that do not appear in the document, a positive real numbers for those that do. 

### Natural-Language Processing (NLP) Pipelines

NLP pipelines pre-process text into a format you can use for classification with a neural net. We'll be doing two things: breaking the dataset into documents, then computing a vocabulary. This will be used for Bag of Words (word count vectors) as well as word vectors, which include context. 

### Vocabulary Computation

A textual dataset is known as a corpus, and the vocabulary of a corpus consists of the unique set of words that the corpus's documents contain. Deeplearning4j has tools to compute vocabulary in different ways. HOW? TK

### Breaking Out the Corpus

In the Kaggle data, each phrase is in a CSV. We need to parse the text into a different form. We do that with a dataset iterator, which gives you a reproducible and testable data pipeline.

### DataSetIterator

When you create a machine-learning model, you need to vectorize the data, because vectors are what machine-learning algorithms understand. 

Vectorization is the process of breaking unstructured data up into a set of features, each of them distinct aspects of the raw data. Each feature is a vector in itself. The two representations we focus on here are word vectors (context, one vector per word which captures its proximity to other words around it) and word count vectors (document level, no context, captures the presence or absence of a word).

When we iterate over a dataset, we retrieve data in a certain order, often from multiple locations. If multiple locations are involved, we want to make the data pipeline retrieval process for each potential end point isolated and testable. We do that with a DataFetcher.

### DataFetcher

As its name implies, a DataFetcher handles data retrieval. Data may live on Amazon Web Services, your local file system, or MySQL. The DataFetcher handles feature vector composition with a process specific to each data source. With this in mind, let's fetch data from a CSV and parse it.

## CSV Processing

Deeplearning4j has a CSV library that allows us to handle CSV values in the following way:

        CSV csv1 = CSV.separator('\t')
                .ignoreLeadingWhiteSpace().skipLines(1)
                .create();
        csv1.read(csv,new CSVReadProc() {
            @Override
            public void procRow(int rowIndex, String... values) {
                
            }
        });

In the above code snippet, CSV is a file object.

The callback lets us access the data. Our goal is to collect the text and create what amounts to a corpus. In the callback, where we're passing in CSVReadProc as an argument to the read method, we want to grab the text and treat each line as a document. Due to the nature of this dataset, we'll just create a list of  documents since it won't use too much memory. 

Since the Kaggle competition is about classifying phrases, we'll be saving each phrase as a row in the list. 

That leads us to a class with an abstract data type called a TextRetriever that contains the following:

    Map<String,Pair<String,String>> //mapping the phraseID to the content and the associated label.

The body of our CSV parser in procRow, cited above, will then be:

    pair.put(values[0],new Pair<>(values[2],values[3]));

That's how we map phrases and labels, and we can use it without coupling it to any particular implementation of our classifier. Now let's test it.

## Testing

Given that we're separating concerns and testing, let's build a unit test. Since our dataset is on our class path, we have a few neat tricks to retrieve it. Deeplearning4j leverages Spring for a few reflection utilities as well as a more robust version of classpath discovery of components. With that in mind our test will looking like the following:

    @Test
    public void testRetrieval() throws  Exception {
         Map<String,Pair<String,String>> data = retriever.data();
        Pair<String,String> phrase2 = data.get("2");
        assertEquals("A series of escapades demonstrating the adage that what is good for the goose",phrase2.getFirst());
        assertEquals("2",phrase2.getSecond());
    }

Our goal is to build a set of abstractions that we can think of as components, and not as individual steps. (One thing we might want to do later is get only phrases, or only labels.) For brevity, I'll let the associated test speak for itself:

    @Test
    public void testNumPhrasesAndLabels() {
        assertEquals(NUM_TRAINING_EXAMPLES,retriever.phrases().size());
        assertEquals(NUM_TRAINING_EXAMPLES,retriever.labels().size());
    }

With many problems, you want to test multiple methods of classification to determine which works best (or use several of them in a so-called ensemble that works better than any one method alone).

## Sentence Iterator

Now that we have basic CSV processing, let's make this into the beginnings of a data pipeline. A sentence iterator will help create a corpus that will lay the groundwork for further processing later. In this particular case, each sentence serves as a "document" in the corpus. 

So what does this corpus look like? 

Our documents have labels, because the data is supervised, and we're implementing a label-aware sentence iterator. The core concept of a sentence iterator is that **it knows where it is in the corpus, will always be able to retrieve the next sentence and can tell us when it's done**. That's a lot of responsibility, so we'll isolate these functions. The core bits of logic are here:
 
    private List<String> phrases;
    private List<String> labels;
    private int currRecord;
    private SentencePreProcessor preProcessor;

Our goal is to keep track of which sentence we're on as well as the current label. We do that with the currRecord position, and we use the text retriever that we built earlier to retrieve the data. This separates the responsibilities of data retrieval and iteration.

Here are the code's juicy parts:

        try {
            TextRetriever retriever = new TextRetriever();
            this.phrases = retriever.phrases();
            this.labels = retriever.labels();
        } catch (IOException e) {
            e.printStackTrace();
        }

Pretty easy. It gives us exactly what we need to iterate over, and we've wrapped it in an interface that's standard across any data pipeline you build with Deeplearning4j.

The test:

    @Test
    public void testIter() {
        LabelAwareSentenceIterator iter = new RottenTomatoesLabelAwareSentenceIterator();
        assertTrue(iter.hasNext());
        //due to the nature of a hashmap internally, may not be the same everytime
        String sentence = iter.nextSentence();
        assertTrue(!sentence.isEmpty());
    }

This will verify that one of our building blocks works, so now we can worry about higher-level concepts like word vectors and Bag of Words. Let's build a BoW DataFetcher first. (It's easier than you think.)

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

Just a few lines, but let's break them down into their components. 

* The iterator. We built it earlier to track where we are currently when iterating over the data. It also associates a string with a dataseet. 
* The count vectorizer. This is the workhorse. Let's load the data in to memory with vectorize and iterate as necessary. 

The process above is **RAM-intensive**, so only run it on a fairly robust server. (TK WHERE ARE THE BENCHMARKS? I'll run these benchmarks for you here.) 

Pruning words from your vocabulary based on TF-IDF gives a good approximation of your data, but we'll skip over that step here. TK WHY? (ND4J only supports dense matrices for the moment, though we're working ways to handle sparse formats.) Since ND4J is a Blas-focused framework, that's what we'll be supporting. 

So what exactly have we done so far? We wrote code to parse CSVs, take the text, map it to labels and iterate through it to produce a matrix. In the process, we built a key component: the vocabulary. It has around 17,000 words. For bag-of-words matrices, this produces a sparse representation of 150,000 rows by 17,000 columns, one column per word.

Bag of Words doesn't give us a lot to work with. After we explore Word2vec, we'll see how the DBN classifier does with both.

## Word Vectors

Let's set *word-count vectors* aside and consider *word vectors*. Remember, word vectors are used to **featurize  textual contexts**. We use the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) with voting on moving windows for document classification.

Since this is a word-vector-based approach, we're going to use Word2vec, and while we're at it, we'll look at how well it trains. Unlike Bag of Words, where features are deterministic (rules based), Word2vec is a form of neural net, which means we're dealing with probabilities and training coefficients. Remember, Word2vec represents word usage, and usage is a matter of probability rather than of lockstep rules.

Since seeing is understanding, we use D3 to visualize the 16,000-word vocabulary. We use an algorithm called t-SNE to gauge the proximity of words to other words. Doing that let's us ensure that the word clusters themselves are coherent.

TK: Add Renders

Word vectors are useful with sequential applications for text. They can be used in document classification with a proper ensemble method (voting) as well with optimizing for a maximum likelihood estimator over the windows and labels.

So what does Word2vec look like in code?

The key code snippet is here:

    Word2vec vec = new Word2Vec.Builder().iterate(iter).tokenizerFactory(factory)
                .learningRate(1e-3).vocabCache(new InMemoryLookupCache(300))
                .layerSize(300).windowSize(5).build();
    vec.fit();

You'll notice we specify a document iterator, a tokenizer factory and a learning rate, among other things. In the second part of this walkthrough, we'll go over these parameters as they apply to a deep-belief network:

* iter: DocumentIterator this is our raw textual pipeline
* factory: our tokenizer factory, handles tokenizing text
* learning rate: step size
* cache: this is where all of our metadata about vocabulary is stored including word vectors, tfidf scores, document frequencies as well as where doucments occurred.
* layer size: this is the number of features per word
* window size: the window size for iterating over text, this is how long of contexts to train on.
