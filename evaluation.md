---
title: Using the Evaluation Classes for Neural Networks
layout: default
---

# Evaluating Performance of Neural Networks Using the Evaluation Classes

Contents

* [Evaluation for Classification](#classification)
* [Evaluation for Regression](#Regression)
* [Time Series Evaluation](#timeseries)
* [Binary Evaluation](#binary)
* [ROC](#roc)
* [Evaluation for Multi-Task Networks](#multitask)



# Overview

When training or deploying a Neural Network it is useful to know the accuracy of your model. In DL4J the Evaluation Class and variants of the Evaluation Class are available to evaluate your model's performance. 



## <a name="classification">Evaluation for Classification</a>


Basic use example of the Evaluation Class. 

Here is an example taken from our dataexamples/CSVExample in the [Examples](https://github.com/deeplearning4j/dl4j-examples) project.

The CSV example has CSV data for 3 classes of flowers and builds a simple feed forward neural network to classify the flowers based on 4 measurements. 

```
Evaluation eval = new Evaluation(3);
INDArray output = model.output(testData.getFeatureMatrix());
eval.eval(testData.getLabels(), output);
log.info(eval.stats());
```

The first line creates an Evaluation object with 3 classes. 
The second line gets the labels from the model for our test dataset. 
The third line uses the eval method to compare the labels array from the testdata with the labels generated from the model. 
The fourth line logs the evaluation data to the console. 

The output.

```
==========================Scores========================================
 Accuracy:        0.9434
 Precision:       0.96
 Recall:          0.9444
 F1 Score:        0.9522
========================================================================
```

By default it displays Accuracy, Precision, Recall and F1 Score. Additionally the Evaluation Class can also display. 

* Confusion Matrix
* False Positive/Negative Rate
* True Positive/Negatice
* Class Count
* and more, see [JavaDoc](https://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html)

Display the Confusion Matrix. 

```
System.out.println(eval.confusionToString());
```

Displays

```
Predicted:         0      1      2
Actual:
0  0          |      16      0      0
1  1          |       0     19      0
2  2          |       0      0     18
```

Additionaly the confusion matrix can be accessed directly, converted to csv or html using.

```
eval.getConfusionMatrix() ;
eval.getConfusionMatrix().toHTML();
eval.getConfusionMatrix().toCSV();
```


## <a name="regression">Evaluation for Regression</a>

To Evaluate a network performing regression use the RegressionEvaluation Class. 

Here is a code snippet with single column, in this case the neural network was predicting the age of shelfish based on measurements. 

```
RegressionEvaluation eval =  new RegressionEvaluation(1);
```

Print the statistics for the Evaluation. 

```
System.out.println(eval.stats());
```

Returns

```
Column    MSE            MAE            RMSE           RSE            R^2            
col_0     7.98925e+00    2.00648e+00    2.82653e+00    5.01481e-01    7.25783e-01    
```

Columns are Mean Squared Error, Mean Absolute Error, Root Mean Squared Error, Relative Squared Error, and R^2 Coefficient of Determination,


## <a name="timeseries">Evaluation for Time Series</a>

Work in Progess


## <a name="binary">Evaluation for Binary Classifiers</a>

[EvaluationBinary](https://deeplearning4j.org/doc/org/deeplearning4j/eval/EvaluationBinary.html)
Is used for evaluating networks with binary classification outputs. The typical classification metrics, such as accuracy, precision, recall, F1 score, etc. are calculated for each output.


## <a name="roc">ROC</a>

[The ROC class](https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROC.html) is used to evaluate Binary Classifiers. Useful methods include: 

* getResults
  * Returns a Results Curve
* calculateAUC
  * Calculates Area Under Curve 

## <a name="multitask">Evaluation for Multi-task Networks</a>

A multi-task network is a network that is trained to produce multiple outputs. For example a network given audio samples can be trained to both predict the language spoken and the gender of the speaker. Multi-task configuration is briefly described [here](https://deeplearning4j.org/compgraph#multitask). 


Evaluation Classes useful for Multi-Task Network

[ROCMultiClass](https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROCMultiClass.html)

[ROCBinary](https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROCBinary.html)




