---
title: 
layout: default
---

# word2vec

One of deep learning's chief applications is in textual analysis, and at the heart of text analysis is [Word2vec](https://code.google.com/p/word2vec/). Word2vec is a neural network that does not implement deep learning, but is crucial to getting input in a numerical form that deep-learning nets can ingest -- the [vector](https://www.khanacademy.org/math/linear-algebra/vectors_and_spaces/vectors/v/vector-introduction-linear-algebra). 

Word2vec creates features without human intervention, and some of those features include the context of individual words; that is, it retains context in the form of multiword windows. In deep learning, the meaning of a word is essentially the words that surround it. Given enough data, usage and context, Word2vec can make highly accurate guesses as to a word’s meaning based on its past appearances. 

The output of the Word2vec neural net is a vocabulary with a vector attached to it, which can be fed into a deep-learning net for classification/labeling. 

There is also a [skip gram representation](http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf) which is used in the dl4j implementation. This has proven to be more accurate due to the more generalizable contexts generated. 

Broadly speaking, we measure words' proximity to each other through their [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), which gauges the similarity between two word vectors. A perfect 90 degree angle represents identity; i.e. France equals France, while Spain has a cosine distance of 0.678515 from France, the highest of any other country.

Here's a graph of words associated with "China" using Word2vec:

![Alt text](../img/word2vec.png)

The other method of preparing text for input to a deep-learning net is called [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BoW). BoW produces a vocabulary with word counts associated to each element of the text. Its output is a wordcount vector. That said, it does not retain context, and therefore is not useful in a granular analysis of those words' meaning. 

# training

Word2Vec trains on raw text. It then records the context, or usage, of each word encoded as word vectors. After training, it's used as lookup table for composition of windows of training text for various tasks in natural-language processing.

Assuming a list of sentences, it's used for lemmatization like this:

         List<String> mySentences = ...;
          
         //source for sentences for word2vec to train on
         SentenceIterator iter = new CollectionSentenceIterator(mySentences);
          
         //tokenization with lemmatization,part of speech taggin,sentence segmentation
         TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
         //use the specified sentence iterator(data source), tokenizer(for vocab), and a min word frequency of 1.
         //Word frequency should be set relative to the size of your dataset.
         Word2Vec vec = new Word2Vec(tokenizerFactory,iter,1);
         vec.train();

From there, Word2vec will do automatic multithreaded training based on your sentence data. After that step, you'll' want to save word2vec like this:

       	 SerializationUtils.saveObject(vec, new File("mypath"));

This will save word2vec to mypath.

You can reload it into memory like this:
        
        Word2Vec vec = SerializationUtils.readObject(new File("mypath"));

From there, you can use Word2vec as a lookup table in the following way:
              
        DoubleMatrix wordVector = vec.getWordVectorMatrix("myword");

        double[] wordVector = vec.getWordVector("myword");

If the word isn't in the vocabulary, Word2vec returns just zeros.


###Windows

Word2Vec works with other neural networks by facilitating the moving window model for training on word occurrences.

It does this via the moving window model. To get windows for text, there are two ways to do this:


                    List<Window> windows = Windows.windows("some text");

This will pull out moving windows of 5 out of the text. Each member of a window is a token.

Another thing you may want to do is use your own custom tokenizer. You can use your own tokenizer by doing the following:


                  TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

                  List<Window> windows = Windows.windows("text",tokenizerFactory);

This will create a tokenizer for the text and create moving windows based on that.


Notably, you can also specify the window size here:
    
                  TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

                  List<Window> windows = Windows.windows("text",tokenizerFactory,windowSize);
     

One other thing that may not be clear is how to train word sequence models. This is done by optimization with the [viterbi algorithm](../doc/org/deeplearning4j/word2vec/viterbi/Viterbi.html).

The general idea behind this is that you train moving windows with word2vec and classify individual windows (with a focus word)

with certain labels. This could be in part of speech tagging, semantic role labeling, named entity recognition, among other tasks.

Viterbi basically calculates the most likely sequence of events (labels) given a transition matrix (the probability of going from
one state to another). Below is an example snippet for setup:

        List<String> labels = ...;
        Index labelIndex = new Index();
		for(int i = 0; i < labels.length; i++)
			labelIndex.add(labels[i]);
		Index featureIndex = ViterbiUtil.featureIndexFromLabelIndex(labelIndex);
		CounterMap<Integer,Integer> transitions = new CounterMap<Integer,Integer>();


This will intialize the baseline features and transitions for viterbi to optimize on.


Say you have the lines of a file with the given transition probabilities given a file:

            //read in the lienes of a file
            List<String> lines = FileUtils.readLines(file);
			for(String line : lines) {
				if(line.isEmpty()) 
					continue;
				List<Window> windows = Windows.windows(line);

				for(int i = 1; i < windows.size(); i++) {
					String firstLabel = windows.get(i - 1).getLabel();
					String secondLabel = windows.get(i).getLabel();
					int first = labelIndex.indexOf(firstLabel);
					int second = labelIndex.indexOf(secondLabel);


					transitions.incrementCount(first,second,1.0);
				}

			}


From there, each line will be something like:

          <ORGANIZATION> IBM </ORGANIZATION> invented a question answering robot called <ROBOT>Watson</ROBOT>.

Given a set of text, Windows.windows automatically infers labels from bracketed capitalized text.


If you do:

             String label = window.getLabel();

on anything containing that window, it will automatically contain that label. This is used in bootstrapping a prior distribution

over the set of labels in a training corpus.

This will save a viteribi implementation for later use:

       
        DoubleMatrix transitionWeights = CounterUtil.convert(transitions);
		Viterbi viterbi = new Viterbi(labelIndex,featureIndex,transitionWeights);
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(output));
		viterbi.write(bos);
		bos.flush();
		bos.close();

