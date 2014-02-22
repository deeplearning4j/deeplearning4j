---
title: 
layout: default
---


# output interpretation


Neural networks can be hard to interpret. You will often ask questions like "How accurate is my model?" or "What do these numbers mean?"

Each kind of neural network has different kinds of output. 

For example, a single-layer network learns to reconstruct input. This is best illustrated by discussing images. The best way to interpret reconstruciton is to see how well an image is reconstructed after the data is "noised" and fed into a neural network. 

As you follow the MNIST tutorial, the best way to gauge your performace is to compare the initial image and network output side by side.

Another example: Multilayer networks are discriminatory classifiers; that is, they label things. 

In machine learning, one metric used to determine how well a classifier performs is called an F1 measure. The F1 measure is a number between zero and one that explains how well the network performed on training. It is analogous to a percentage, with 1 the equivalent of 100 percent.

DL4J has a class called [Evaluation](../doc/org/deeplearning4j/eval/Evaluation.html) that will output F1 for you.

Here's one way to use it: 

         
         BaseMultiLayerNetwork network = ....;

         DoubleMatrix inputToPredict = ...;

         DoubleMatrix actualLabels = ...;

         DoubleMatrix predicted = network.predict(inputToPredict);

         Evaluation eval = new Evaluation();

         eval.eval(actualLabels,predicted);


         System.out.println(eval.stats());



Each label is a binary matrix of 1 row and, say, 10 columns, with each column representing a number from one to 10. (The number of columns will vary with the number of possible outcomes, or labels.) There can only be a single 1 in this matrix, and it is located in the column representing the number labeled. That is, a [0 1 0 0 0 0 0 0 0 0] is two, and so forth. 

Each label is then assigned a likelihood of how accurately it describes the input, according to the features recognized by your network. Those probabilities are the network's guesses. At the end of your test, you compare the highest-probability label with the actual number that was the input. The aggregate of these comparisons is your accuracy rate, or f score. 

In fact, you can enter any number of inputs at once. Each of them will be a row in your binary matrix. And the number of rows in your binary input matrix will be equal to the number of rows in your binary out matrix of guesses.

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