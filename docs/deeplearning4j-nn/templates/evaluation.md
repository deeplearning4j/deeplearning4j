---
title: Evaluation Classes for Neural Networks
short_title: Evaluation
description: Tools and classes for evaluating neural network performance
category: Tuning & Training
weight: 3
---


## Why evaluate?

When training or deploying a Neural Network it is useful to know the accuracy of your model. In DL4J the Evaluation Class and variants of the Evaluation Class are available to evaluate your model's performance. 


## <a name="classification">Evaluation for Classification</a>


The Evaluation class is used to evaluate the performance for binary and multi-class classifiers (including time series classifiers). This section covers basic usage of the Evaluation Class.

Given a dataset in the form of a DataSetIterator, the easiest way to perform evaluation is to use the built-in evaluate methods on MultiLayerNetwork and ComputationGraph:
```
DataSetIterator myTestData = ...
Evaluation eval = model.evaluate(myTestData);
```

However, evaluation can be performed on individual minibatches also. Here is an example taken from our dataexamples/CSVExample in the [Examples](https://github.com/deeplearning4j/dl4j-examples) project.

The CSV example has CSV data for 3 classes of flowers and builds a simple feed forward neural network to classify the flowers based on 4 measurements. 

```
Evaluation eval = new Evaluation(3);
INDArray output = model.output(testData.getFeatures());
eval.eval(testData.getLabels(), output);
log.info(eval.stats());
```

The first line creates an Evaluation object with 3 classes. 
The second line gets the labels from the model for our test dataset. 
The third line uses the eval method to compare the labels array from the testdata with the labels generated from the model. 
The fourth line logs the evaluation data to the console. 

The output.

```
Examples labeled as 0 classified by model as 0: 24 times
Examples labeled as 1 classified by model as 1: 11 times
Examples labeled as 1 classified by model as 2: 1 times
Examples labeled as 2 classified by model as 2: 17 times


==========================Scores========================================
 # of classes:    3
 Accuracy:        0.9811
 Precision:       0.9815
 Recall:          0.9722
 F1 Score:        0.9760
Precision, recall & F1: macro-averaged (equally weighted avg. of 3 classes)
========================================================================
```

By default the .stats() method displays the confusion matrix entries (one per line), Accuracy, Precision, Recall and F1 Score. Additionally the Evaluation Class can also calculate and return the following values:

* Confusion Matrix
* False Positive/Negative Rate
* True Positive/Negative
* Class Counts
* F-beta, G-measure, Matthews Correlation Coefficient and more, see [Evaluation JavaDoc](https://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html)

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

As with the Evaluation class, RegressionEvaluation on a DataSetIterator can be performed as follows:
```
DataSetIterator myTestData = ...
RegressionEvaluation eval = model.evaluateRegression(myTestData);
```

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

Columns are Mean Squared Error, Mean Absolute Error, Root Mean Squared Error, Relative Squared Error, and R^2 Coefficient of Determination

See [RegressionEvaluation JavaDoc](https://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/eval/RegressionEvaluation.html)

## <a name="multiple">Performing Multiple Evaluations Simultaneously</a>

When performing multiple types of evaluations (for example, Evaluation and ROC on the same network and dataset) it is more efficient to do this in one pass of the dataset, as follows:

```
DataSetIterator testData = ...
Evaluation eval = new Evaluation();
ROC roc = new ROC();
model.doEvaluation(testdata, eval, roc);
```

## <a name="timeseries">Evaluation of Time Series</a>

Time series evaluation is very similar to the above evaluation approaches. Evaluation in DL4J is performed on all (non-masked) time steps separately - for example, a time series of length 10 will contribute 10 predictions/labels to an Evaluation object.
One difference with time seires is the (optional) presence of mask arrays, which are used to mark some time steps as missing or not present. See [Using RNNs - Masking](./deeplearning4j-nn-recurrent) for more details on masking.

For most users, it is simply sufficient to use the ```MultiLayerNetwork.evaluate(DataSetIterator)``` or ```MultiLayerNetwork.evaluateRegression(DataSetIterator)``` and similar methods. These methods will properly handle masking, if mask arrays are present.


## <a name="binary">Evaluation for Binary Classifiers</a>

The EvaluationBinary is used for evaluating networks with binary classification outputs - these networks usually have Sigmoid activation functions and XENT loss functions. The typical classification metrics, such as accuracy, precision, recall, F1 score, etc. are calculated for each output.

```
EvaluationBinary eval = new EvaluationBinary(int size)
```

See [EvaluationBinary JavaDoc](https://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/eval/EvaluationBinary.html)


## <a name="roc">ROC</a>

ROC (Receiver Operating Characteristic) is another commonly used evaluation metric for the evaluation of classifiers. Three ROC variants exist in DL4J:

- ROC - for single binary label (as a single column probability, or 2 column 'softmax' probability distribution).
- ROCBinary - for multiple binary labels
- ROCMultiClass - for evaluation of non-binary classifiers, using a "one vs. all" approach 

These classes have the ability to calculate the area under ROC curve (AUROC) and area under Precision-Recall curve (AUPRC), via the ```calculateAUC()``` and ```calculateAUPRC()``` methods. Furthermore, the ROC and Precision-Recall curves can be obtained using ```getRocCurve()``` and ```getPrecisionRecallCurve()```.

The ROC and Precision-Recall curves can be exported to HTML for viewing using: ```EvaluationTools.exportRocChartsToHtmlFile(ROC, File)```, which will export a HTML file with both ROC and P-R curves, that can be viewed in a browser.


Note that all three support two modes of operation/calculation
- Thresholded (approximate AUROC/AUPRC calculation, no memory issues)
- Exact (exact AUROC/AUPRC calculation, but can require large amount of memory with very large datasets - i.e., datasets with many millions of examples)

The number of bins can be set using the constructors. Exact can be set using the default constructor ```new ROC()``` or explicitly using ```new ROC(0)```

See [ROCBinary JavaDoc](https://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/eval/ROC.html) is used to evaluate Binary Classifiers.

## <a name="calibration">Evaluating Classifier Calibration</a>

Deeplearning4j also has the EvaluationCalibration class, which is designed to analyze the calibration of a classifier. It provides a number of tools for this purpose:
 
 - Counts of the number of labels and predictions for each class
 - Reliability diagram (or reliability curve)
 - Residual plot (histogram)
 - Histograms of probabilities, including probabilities for each class separately
 
 Evaluation of a classifier using EvaluationCalibration is performed in a similar manner to the other evaluation classes.
 The various plots/histograms can be exported to HTML for viewing using ```EvaluationTools.exportevaluationCalibrationToHtmlFile(EvaluationCalibration, File)```.

## <a name="spark">Distributed Evaluation for Spark Networks</a>

SparkDl4jMultiLayer and SparkComputationGraph both have similar methods for evaluation:
```
Evaluation eval = SparkDl4jMultiLayer.evaluate(JavaRDD<DataSet>);

//Multiple evaluations in one pass:
SparkDl4jMultiLayer.doEvaluation(JavaRDD<DataSet>, IEvaluation...);
```


## <a name="multitask">Evaluation for Multi-task Networks</a>

A multi-task network is a network that is trained to produce multiple outputs. For example a network given audio samples can be trained to both predict the language spoken and the gender of the speaker. Multi-task configuration is briefly described [here](./deeplearning4j-nn-computationgraph). 

Evaluation Classes useful for Multi-Task Network

See [ROCMultiClass JavaDoc](https://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/eval/ROCMultiClass.html)

See [ROCBinary JavaDoc](https://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/eval/ROCBinary.html)