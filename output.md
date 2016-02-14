---
title: Output Interpretation
layout: default
---

# Output Interpretation

The output of neural networks can be hard to interpret. You'll probably ask yourself questions like "How accurate is my model?" or "What do these numbers mean?" Not only that, but each kind of neural network has a different kind of output. 

For example, a single-layer network learns to reconstruct input. This is best illustrated by discussing images. The best way to interpret reconstruciton is to see how well an image is reconstructed after the data is "noised" and fed into a neural network. 

As you follow the MNIST tutorial, the best way to gauge your performace is to compare the initial image and network output side by side.

Another example would be multilayer networks. Multilayer networks are discriminatory classifiers; that is, they label things. 

In machine learning, one metric used to determine how well a classifier performs is called the f1 score. The f1 score is a number between zero and one that explains how well the network performed during training. It is analogous to a percentage, with 1 being the equivalent of 100 percent predictive accuracy.

DL4J has a class called [Evaluation](../doc/org/deeplearning4j/eval/Evaluation.html) that will output f1 scores for you.

Imagine each label is a binary matrix of 1 row and, say, 10 columns, with each column representing a number from one to 10. (The number of columns will actually vary with the number of possible outcomes, or labels.) There can only be a single 1 in this matrix, and it is located in the column representing the number labeled. That is, [0 1 0 0 0 0 0 0 0 0] means two, and so forth. 
Each label is then assigned a likelihood of how accurately it describes the input, according to the features recognized by your network. Those probabilities are the network's guesses. At the end of your test, you compare the highest-probability label with the actual number of the input. The aggregate of these comparisons is your accuracy rate, or f score. 

In fact, you can enter any number of inputs into the network simultaneously. Each of them will be a row in your binary matrix. And the number of rows in your binary input matrix will be equal to the number of rows in your binary out matrix of guesses.

You can also create one evaluation class to track statitistics over time. Using a data set iterator, you could do something like this:

            DataSetIterator iter = ...;

            while(iter.hasNext()) {
              DataSet next = iter.next();
              DoubleMatrix predicted = network.predict(next.getFirst());
              DoubleMatrix actual = next.getSecond();
              eval.eval(acutal,predicted);

            }

            System.out.println(eval.stats());

This will allow you to iterate over a data set and cumulatively add to the results. The eval.stats() call will print the confusion matrices and f scores at the bottom. Those scores are good indicators of performance. 
