---
layout: default
---

# custom data sets 

Garbage in, garbage out. 

Deep learning, and machine learning more generally, needs a good training set to work properly. Collecting and constructing the training set -- a sizable body of known data -- takes time, domain-specific knowledge of where and how to gather relevant information. The training set acts as the benchmark against which deep-learning nets are trained. That is what they learn to reconstruct when they are eventually unleashed on unstructured data. 

At this stage, lengthy human intervention is necessary to find the right raw data and transform it into a numerical representation that the algorithm can understand. Building a training set is, in a sense, pre-pre-training. For a text-based training set, you may even have to do some feature creation. 

Training sets that require much time or expertise can serve as a proprietary edge in the competitive world of data science and problem solving. The nature of the expertise is largely in telling your algorithm what matters to you through the training set. 

It involves telling a story -- through the initial data you select -- that will guide your deep-learning nets as they extrapolate the significant features, both in the training set and in the unstructured data they've been created to study.

To create a useful training set, you have to understand the problem you're solving; i.e. what you want your deep-learning nets to pay attention to. 

All input to the deep-learning nets -- whether it's words, images or other data -- must be transformed into numbers known as vectors, in a process called vectorization. A vector is simply a one-column matrix with an extendible number of rows.

Vectorization is done via the [../doc/org/deeplearning4j/datasets/iterator/DataSetIterator.html](DataSetIterator) . 

A [../doc/org/deeplearning4j/datasets/iterator/DataSetIterator.html](DataSetIterator) is capable of iterating through a data set 

and vectorizing it in to the apporpriate format.

Extending  [../doc/org/deeplearning4j/datasets/iterator/BaseDataSetIterator.html](BaseDataSetIterator) you get the baseline functionality

of batching and data set inputs. The constructor below represents taking in a batch size and a number of inputs. For example on mnist,

there are 60k images and we want them in batches of 10, the constructor for this is the following:


                            new MnistDataSetIterator(10,60000)


That constructor is inherited from the BaseDataSetIterator. Underneath, the BaseDataSetIterator relies on a 

 [../doc/org/deeplearning4j/datasets/iterator/DataSetFetcher.html](DataSetFetcher)


 The  [../doc/org/deeplearning4j/datasets/iterator/DataSetFetcher.html](DataSetFetcher)  handles vectorization of input data in to a format suitable for consumption by a neural network.


 A  [../doc/org/deeplearning4j/datasets/iterator/DataSetFetcher.html](DataSetFetcher) is used as follows:


                     DataSetFetcher fetcher = ...;
                     //fetch n examples
                     fetcher.fetch(numExamples);

                     DataSet myData = fetcher.next();


 This returns the dataset for neural network consumption. The call to fetch tells the fetcher to get the next n examples from your input source.


 Following that, we can extend something called a [../doc/org/deeplearning4j/datasets/fetchers/BaseDataFetcher.html](BaseDataFetcher) .

 This provides a few baseline methods for converting say: output labels to an output matrix, This also includes baseline methods

 to think about when fetching data from an input source.



 Below is how to input different kinds of data in to a neural network.



 Images:

       With images, you will typically transform load the image. This can be done with an  [../doc/org/deeplearning4j/datasets/vectorizer/ImageVectorizer.html](ImageVectorizer) . An ImageVectorizer loads in the image from a file and takes the image and transforms its pixels based on the RGB spectrum.

 Text:

          With text, there are 2 ways of transforming textual data in to something a neural network understands.

          One is the bag of words approach, which takes in the whole corpus of text, and calculates a vocab.

          Any document is then represented as a bag of words, or a based on the vocab each word is a column with the word counts in it.


          The other approach is via [../doc/org/deeplearning4j/word2vec/Word2Vec.html]. Word2Vec takes in to account the distributional context of a word

          and learns word vectors. These vectors then are concacneated in to a window vector. 

          If we think about a word window:


                w1 w2 w3

         Each word vector is taken from word2vec and combined in to a singular row vector which becomes a representation of the window.