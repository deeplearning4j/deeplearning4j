Output Interpretation
=====================================


Neural networks can be hard to interpret. You will often ask questions such as: "How accurate is my model?" or "What do these numbers mean?"


Each kind of neural network has different kinds of output. A single layered network learns to reconstruct input.

Due to the nature of the problem, this is best described with images. The best way for interpreting reconstruciton is

to see how well an image is reconstructed after being fed in to a neural network. Following the tutorial from MNIST,

the best way would be to compare the images side by side and visualize the results.

Multi layer networks are discriminatory classifiers, aka they label things. In Machine Learning, one metric 

we use to determine how well a classifier performs is called an F1 measure. 

This is a number on a range from zero to one (think percentages) that explains how well the network performed on training.

DL4J has a class called [Evaluation](../doc/org/deeplearning4j/eval/Evaluation.html) that will output this number for you.

An example usage is follows:

         
         BaseMultiLayerNetwork network = ....;

         DoubleMatrix inputToPredict = ...;

         DoubleMatrix actualLabels = ...;

         DoubleMatrix predicted = network.predict(inputToPredict);

         Evaluation eval = new Evaluation();

         eval.eval(actualLabels,predicted);


         System.out.println(eval.stats());



This basic example takes the actual labels(a binary matrix where 1 is the label) and the guesses (probabilies of each label)

and compares them against each other to match the label. You can create one evaluation class and use that to track statitistics over time.

One will want to usually predict lots of data. Using a data set iterator, we could do soemthing like this:



            DataSetIterator iter = ...;



            while(iter.hasNext()) {
              DataSet next = iter.next();
              DoubleMatrix predicted = network.predict(next.getFirst());
              DoubleMatrix actual = next.getSecond();
              eval.eval(acutal,predicted);

            }


            System.out.println(eval.stats());



This will allow you to iterate over a data set and cumulatively add to the results. The eval.stats() call will print the confusion matrices and f scores at the very bottom.

This is a good indicator of performance.


