---
layout: default
---

# custom data sets 

All input to the deep-learning nets -- whether it's words, images or other data -- must be transformed into numbers known as vectors, in a process called vectorization. A vector is simply a one-column matrix with an extendible number of rows.

### vectorization

Vectorization is done with the [DataSetIterator](../doc/org/deeplearning4j/datasets/iterator/DataSetIterator.html), which can iterate over data sets and vectorize them; i.e. translate them to a machine-friendly numerical form. It knows how to iterate through a data set and load it in to a [DataSet](../doc/org/deeplearning4j/datasets/DataSet.html).

(Optimal batch size from Datasetiterator is n x m, where n is the minibatch size and m is the number of workers on your distriputed network.)

Extending a [BaseDataSetIterator](../doc/org/deeplearning4j/datasets/iterator/BaseDataSetIterator.html) lets you do batching and data set inputs. The constructor below takes two parameters: the data set to be iterated over, and the batch size of each iteration. For example, on MNIST, there are 60,000 images, and we'll handle them here in batches of 10, with this command:


         new MnistDataSetIterator(10,60000)

The constructor above inherits from the BaseDataSetIterator, which itself relies on a [DataSetFetcher](../doc/org/deeplearning4j/datasets/iterator/DataSetFetcher.html). The DataSetFetcher is called by the iterator to vectorize the input data. A DataSetFetcher knows how to load data from images or text. It has an API similar to the data set iterator.

Here's how you use it:

         DataSetFetcher fetcher = ...;
         //fetch n examples
         fetcher.fetch(numExamples);

         DataSet myData = fetcher.next();

This returns the dataset for consumption by the neural network. The fetch call tells the fetcher to get the next n examples from your input source.

After that, we can extend something called a [BaseDataFetcher](../doc/org/deeplearning4j/datasets/fetchers/BaseDataFetcher.html). This provides a few baseline methods for converting, say, output labels to an output matrix. It also includes a few methods to think about when fetching data from an input source. Below, we'll explore different kinds of data you can feed into a neural network.

### images

With images, you typically transform load the image. This can be done with an [ImageVectorizer](../doc/org/deeplearning4j/datasets/vectorizer/ImageVectorizer.html), which loads the image from a file and transforms its pixels based on the RGB spectrum.

Note that the ImageVectorizer takes in a label number. You typically want a set of images in a folder named after their label. If you're doing digits with MNIST, you're label files might look like this:
                         
        parentdir/
             1/
              img1.png
              img2.png
             2/
              img3.png
              img4.png

Given an image data set where the labels are child directories, you could do something like:

         File rootDir = new File("path/to/your/dir");
         //needs to be a list for maintaining order of labels
         List<String> labels = new ArrayList<String>();

         for(File f : rootDir.listFiles()) {
            if(f.isDirectory())
         	labels.add(f.getName());
         }

When you instantiate the ImageVectorizer, you could do something like this:

           

         File yourImage = new File("path/to/your/file");
         Vectorizer v = new ImageVectorizer(,labels.size(),labels.indexOf(yourImage.getParentFile().getName()));
         DataSet d = v.vectorize();

### text

There are two ways to transform textual data into forms a neural network understands.

The first is the bag of words (BoW) approach, which ingests the corpus of text, determines the vocabulary, and associates a word count with each lexical unit. Any document is then represented as a so-called "bag of words," which is nothing more than a column of extendible rows, each one containing the word count for one lexical unit. BoW is useful in topic modelling and document classification. It primarily answers the question: What is this text about? 

The other approach is [Word2Vec](../doc/org/deeplearning4j/word2vec/Word2Vec.html), which takes into account the distributional context of a word and learns word vectors. A word vector is a series of numbers associated with one word. 

Words are then grouped in windows of varying length. [Barack Obama], for example, is a word window with a length of two; [the United States of America] is a word window with a length of five. Each word's vector is concatenated into the window vector.

Let's take a word window of size three:

        w1 w2 w3

and assume that w1's vector is [1 2 3], w2's vector is [4 5 6] and w3's is [7 8 9].

Each word vector is taken from Word2vec and combined in to a singular row vector which becomes a representation of the window. In this case, the window vector would be [1 2 3 4 5 6 7 8 9].

The neural net known as Word2vec is useful for named-entity recognition, semantic role labeling, summarization, lemmatization, parts-of-speech tagging, question and answer, and relationship extraction. If BoW is macro, Word2vec is micro. It's primarily concerned with questions about elements of the text rather than the text in its entirety.


In our case, we use [Word2Vec](../doc/org/deeplearning4j/word2vec/Word2Vec.html) to input word contexts in to a neural network for training. Each context has an associated label.

An example is as follows:
        
                String text = "some text";
                Word2Vec vec = ...;
                List<Window> windows = Windows.windows(text);
                for(Window window : windows) 
                {
                  //eample for the given window based on the vocab in word2vec
                  double[] example = WindowConverter.asExample(window,vec);

                }

Working with your own datasets will present complications, but it will also show you the real power of deep learning. Deep learning works with sets of images and symbols, the latter including text, numbers and genetic code. Next, we'll explore how deep learning can be useful in [textual analysis](../textanalysis.html). 