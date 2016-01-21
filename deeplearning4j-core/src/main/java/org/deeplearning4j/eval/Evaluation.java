/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.eval;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.*;

import org.deeplearning4j.berkeley.Counter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Evaluation metrics: precision, recall, f1
 * @author Adam Gibson
 *
 */
public class Evaluation<T extends Comparable<? super T>> implements Serializable {

    protected Counter<Integer> truePositives = new Counter<>();
    protected Counter<Integer> falsePositives = new Counter<>();
    protected Counter<Integer> trueNegatives = new Counter<>();
    protected Counter<Integer> falseNegatives = new Counter<>();
    protected ConfusionMatrix<Integer> confusion;
    protected int numRowCounter = 0;
    protected List<Integer> labelsList = new ArrayList<>();
    protected Map<Integer, String> labelsMap = new HashMap<>();
    protected static Logger log = LoggerFactory.getLogger(Evaluation.class);
    //What to output from the precision/recall function when we encounter an edge case
    protected static final double DEFAULT_EDGE_VALUE = 0.0;

    // Empty constructor
    public Evaluation() {}

    // Constructor that takes number of output classes
    public Evaluation(int numClasses) {
        for(int i = 0; i < numClasses; i++)
            labelsList.add(i);
        confusion = new ConfusionMatrix<>(labelsList);
    }

    public Evaluation(List<String> labels) {
        if(labels != null && !labels.isEmpty()) {
            int i = 0;
            for (String label : labels) {
                this.labelsMap.put(i, label);
                i++;
            }
        }
    }

    public Evaluation(Map<Integer, String> labels) {
        this.labelsMap = labels;
    }

    /**
     * Collects statistics on the real outcomes vs the
     * guesses. This is for logistic outcome matrices.
     *
     * Note that an IllegalArgumentException is thrown if the two passed in
     * matrices aren't the same length.
     * @param realOutcomes the real outcomes (labels - usually binary)
     * @param guesses the guesses/prediction (usually a probability vector)
     * */
    public void eval(INDArray realOutcomes,INDArray guesses) {
        // Add the number of rows to numRowCounter
        numRowCounter += realOutcomes.shape()[0];

        // If confusion is null, then Evaluation is instantiated without providing the classes
        if(confusion == null) {
            log.warn("Creating confusion matrix based on classes passed in . Will assume the label distribution passed in is indicative of the overall dataset");
            Set<Integer> classes = new HashSet<>();
            // Infer all the class label based on mini batch
            for(int i = 0; i < realOutcomes.columns(); i++) {
                classes.add(i);
            }
            // Create confusion matrix based on potentially incomplete set of labels
            confusion = new ConfusionMatrix<>(new ArrayList<>(classes));
        }

        // Length of real labels must be same as length of predicted labels
        if(realOutcomes.length() != guesses.length())
            throw new IllegalArgumentException("Unable to evaluate. Outcome matrices not same length");

        // For each row get the most probable label (column) from prediction and assign as guessMax
        // For each row get the column of the true label and assign as currMax
        for(int i = 0; i < realOutcomes.rows(); i++) {
            INDArray currRow = realOutcomes.getRow(i);
            INDArray guessRow = guesses.getRow(i);

            int currMax;
            {
                double max = currRow.getDouble(0);
                currMax = 0;
                for (int col = 1; col < currRow.columns(); col++) {
                    if (currRow.getDouble(col) > max) {
                        max = currRow.getDouble(col);
                        currMax = col;
                    }
                }
            }
            int guessMax;
            {
                double max = guessRow.getDouble(0);
                guessMax = 0;
                for (int col = 1; col < guessRow.columns(); col++) {
                    if (guessRow.getDouble(col) > max) {
                        max = guessRow.getDouble(col);
                        guessMax = col;
                    }
                }
            }

            // Add to the confusion matrix the real class of the row and
            // the predicted class of the row
            addToConfusion(currMax, guessMax);

            // If they are equal
            if(currMax == guessMax) {
                // Then add 1 to True Positive
                // (For a particular label)
                incrementTruePositives(guessMax);

                // And add 1 for each negative class that is accurately predicted (True Negative)
                //(For a particular label)
                for(Integer clazz : confusion.getClasses()) {
                    if(clazz != guessMax)
                      trueNegatives.incrementCount(clazz, 1.0);
               }
            }
            else {
                // Otherwise the real label is predicted as negative (False Negative)
                incrementFalseNegatives(currMax);
                // Otherwise the prediction is predicted as falsely positive (False Positive)
                incrementFalsePositives(guessMax);
                // Otherwise true negatives
                for (Integer clazz : confusion.getClasses()) {
                    if (clazz != guessMax && clazz != currMax)
                        trueNegatives.incrementCount(clazz, 1.0);

                }
            }
        }
    }

    /**
     * Convenience method for evaluation of time series.
     * Reshapes time series (3d) to 2d, then calls eval
     * @see #eval(INDArray, INDArray)
     */
    public void evalTimeSeries(INDArray labels, INDArray predicted){
        if(labels.rank() == 2 && predicted.rank() == 2) eval(labels,predicted);
        if(labels.rank() != 3 ) throw new IllegalArgumentException("Invalid input: labels are not rank 3 (rank="+labels.rank()+")");
        if(!Arrays.equals(labels.shape(),predicted.shape())){
            throw new IllegalArgumentException("Labels and predicted have different shapes: labels="
                + Arrays.toString(labels.shape()) + ", predicted="+Arrays.toString(predicted.shape()));
        }

        if( labels.ordering() == 'f' ) labels = Shape.toOffsetZeroCopy(labels, 'c');
        if( predicted.ordering() == 'f' ) predicted = Shape.toOffsetZeroCopy(predicted, 'c');

        //Reshape, as per RnnToFeedForwardPreProcessor:
        int[] shape = labels.shape();
        labels = labels.permute(0,2,1);	//Permute, so we get correct order after reshaping
        labels = labels.reshape(shape[0] * shape[2], shape[1]);

        predicted = predicted.permute(0, 2, 1);
        predicted = predicted.reshape(shape[0] * shape[2], shape[1]);

        eval(labels,predicted);
    }

    /**
     * Evaluate a time series, whether the output is masked usind a masking array. That is,
     * the mask array specified whether the output at a given time step is actually present, or whether it
     * is just padding.<br>
     * For example, for N examples, nOut output size, and T time series length:
     * labels and predicted will have shape [N,nOut,T], and outputMask will have shape [N,T].
     * @see #evalTimeSeries(INDArray, INDArray)
     */
    public void evalTimeSeries(INDArray labels, INDArray predicted, INDArray outputMask){

        int totalOutputExamples = outputMask.sumNumber().intValue();
        int outSize = labels.size(1);

        INDArray labels2d = Nd4j.create(totalOutputExamples, outSize);
        INDArray predicted2d = Nd4j.create(totalOutputExamples,outSize);

        int rowCount = 0;
        for( int ex=0; ex<outputMask.size(0); ex++ ){
            for( int t=0; t<outputMask.size(1); t++ ){
                if(outputMask.getDouble(ex,t) == 0.0) continue;

                labels2d.putRow(rowCount, labels.get(NDArrayIndex.point(ex), NDArrayIndex.all(), NDArrayIndex.point(t)));
                predicted2d.putRow(rowCount, predicted.get(NDArrayIndex.point(ex), NDArrayIndex.all(), NDArrayIndex.point(t)));

                rowCount++;
            }
        }
        eval(labels2d,predicted2d);
    }

    /**
     * Evaluate a single prediction (one prediction at a time)
     * @param predictedIdx Index of class predicted by the network
     * @param actualIdx Index of actual class
     */
    public void eval(int predictedIdx, int actualIdx ) {
        // Add the number of rows to numRowCounter
        numRowCounter++;

        // If confusion is null, then Evaluation is instantiated without providing the classes
        if(confusion == null) {
            throw new UnsupportedOperationException("Cannot evaluate single example without initializing confusion matrix first");
        }

        addToConfusion(predictedIdx, actualIdx);

        // If they are equal
        if(predictedIdx == actualIdx) {
            // Then add 1 to True Positive
            // (For a particular label)
            incrementTruePositives(predictedIdx);

            // And add 1 for each negative class that is accurately predicted (True Negative)
            //(For a particular label)
            for(Integer clazz : confusion.getClasses()) {
                if(clazz != predictedIdx)
                    trueNegatives.incrementCount(clazz, 1.0);
            }
        }
        else {
            // Otherwise the real label is predicted as negative (False Negative)
            incrementFalseNegatives(actualIdx);
            // Otherwise the prediction is predicted as falsely positive (False Positive)
            incrementFalsePositives(predictedIdx);
            // Otherwise true negatives
            for (Integer clazz : confusion.getClasses()) {
                if (clazz != predictedIdx && clazz != actualIdx)
                    trueNegatives.incrementCount(clazz, 1.0);

            }
        }
    }

    public String stats() {
        return stats(true);
    }

    /**
     * Method to obtain the classification report as a String
     *
     * @param suppressWarnings whether or not to output warnings related to the evaluation results
     * @return A (multi-line) String with accuracy, precision, recall, f1 score etc
     */
    public String stats(boolean suppressWarnings) {
        String actual, expected;
        StringBuilder builder = new StringBuilder().append("\n");
        StringBuilder warnings = new StringBuilder();
        List<Integer> classes = confusion.getClasses();
        for (Integer clazz : classes) {
            actual = resolveLabelForClass(clazz);
            //Output confusion matrix
            for (Integer clazz2 : classes) {
                int count = confusion.getCount(clazz, clazz2);
                if (count != 0) {
                    expected = resolveLabelForClass(clazz2);
                    builder.append(String.format("Examples labeled as %s classified by model as %s: %d times\n", actual, expected, count));
                }
            }

            //Output possible warnings regarding precision/recall calculation
            if (!suppressWarnings && truePositives.getCount(clazz) == 0) {
                if (falsePositives.getCount(clazz) == 0) {
                    warnings.append(String.format("Warning: class %s was never predicted by the model. This class was excluded from the average precision\n", actual));
                }
                if (falseNegatives.getCount(clazz) == 0) {
                    warnings.append(String.format("Warning: class %s has never appeared as a true label. This class was excluded from the average recall\n", actual));
                }
            }
        }
        builder.append("\n");
        builder.append(warnings);

        DecimalFormat df = new DecimalFormat("#.####");
        builder.append("\n==========================Scores========================================");
        builder.append("\n Accuracy:  ").append(df.format(accuracy()));
        builder.append("\n Precision: ").append(df.format(precision()));
        builder.append("\n Recall:    ").append(df.format(recall()));
        builder.append("\n F1 Score:  ").append(df.format(f1()));
        builder.append("\n========================================================================");
        return builder.toString();
    }

    private String resolveLabelForClass(Integer clazz) {
        //Use label in map if it is correct, integer otherwise
        String label = labelsMap.get(clazz);
        if (label == null || label.isEmpty()) {
            label = clazz.toString();
        }
        return label;
    }

    /**
     * Returns the precision for a given label
     * @param classLabel the label
     * @return the precision for the label
     */
    public double precision(Integer classLabel) {
        return precision(classLabel, DEFAULT_EDGE_VALUE);
    }

    /**
     * Returns the precision for a given label
     * @param classLabel the label
     * @param edgeCase What to output in case of 0/0
     * @return the precision for the label
     */
    public double precision(Integer classLabel, double edgeCase) {
        double tpCount = truePositives.getCount(classLabel);
        double fpCount = falsePositives.getCount(classLabel);

        //Edge case
        if (tpCount == 0 && fpCount == 0) {
            return edgeCase;
        }

        return tpCount / (tpCount + fpCount);
    }

    /**
     * Precision based on guesses so far
     * Takes into account all known classes and outputs average precision across all of them
     * @return the total precision based on guesses so far
     */
    public double precision() {
        double precisionAcc = 0.0;
        int classCount = 0;
        for(Integer classLabel : confusion.getClasses()) {
            double precision = precision(classLabel, -1);
            if (precision != -1) {
                precisionAcc += precision(classLabel);
                classCount++;
            }
        }
        return precisionAcc / (double) classCount;
    }

    /**
     * Returns the recall for a given label
     * @param classLabel the label
     * @return Recall rate as a double
     */
    public double recall(Integer classLabel) {
        return recall(classLabel, DEFAULT_EDGE_VALUE);
    }

    /**
     * Returns the recall for a given label
     * @param classLabel the label
     * @param edgeCase What to output in case of 0/0
     * @return Recall rate as a double
     */
    public double recall(Integer classLabel, double edgeCase) {
        double tpCount = truePositives.getCount(classLabel);
        double fnCount = falseNegatives.getCount(classLabel);

        //Edge case
        if (tpCount == 0 && fnCount == 0) {
            return edgeCase;
        }

        return tpCount / (tpCount + fnCount);
    }

    /**
     * Recall based on guesses so far
     * Takes into account all known classes and outputs average recall across all of them
     * @return the recall for the outcomes
     */
    public double recall() {
        double recallAcc = 0.0;
        int classCount = 0;
        for(Integer classLabel : confusion.getClasses()) {
            double recall = recall(classLabel, -1.0);
            if (recall != -1.0) {
                recallAcc += recall(classLabel);
                classCount++;
            }
        }
        return recallAcc / (double) classCount;
    }

    /**
     * Calculate f1 score for a given class
     * @param classLabel the label to calculate f1 for
     * @return the f1 score for the given label
     */
    public double f1(Integer classLabel) {
        double precision = precision(classLabel);
        double recall = recall(classLabel);
        if(precision == 0 || recall == 0)
            return 0;
        return 2.0 * ((precision * recall / (precision + recall)));
    }

    /**
     * TP: true positive
     * FP: False Positive
     * FN: False Negative
     * F1 score: 2 * TP / (2TP + FP + FN)
     * @return the f1 score or harmonic mean based on current guesses
     */
    public double f1() {
        double precision = precision();
        double recall = recall();
        if(precision == 0 || recall == 0)
            return 0;
        return 2.0 * ((precision * recall / (precision + recall)));
    }

    /**
     * Accuracy:
     * (TP + TN) / (P + N)
     * @return the accuracy of the guesses so far
     */
    public double accuracy() {
        return truePositives() / getNumRowCounter();
    }


    // Access counter methods
    /**
     * True positives: correctly rejected
     * @return the total true positives so far
     */
    public double truePositives() {
        return truePositives.totalCount();
    }
    /**
     * True negatives: correctly rejected
     * @return the total true negatives so far
     */
    public double trueNegatives() {
        return trueNegatives.totalCount();
    }
    /**
     * False positive: wrong guess
     * @return the count of the false positives
     */
    public double falsePositives() {
        return falsePositives.totalCount();
    }
    /**
     * False negatives: correctly rejected
     * @return the total false negatives so far
     */
    public double falseNegatives() {
        return falseNegatives.totalCount();
    }
    /**
     * Total negatives true negatives + false positives
     * @return the overall negative count
     */
    public double negative() {
        return trueNegatives() + falsePositives();
    }
    /**
     * Returns all of the positive guesses:
     * true positive + false negative
     * @return
     */
    public double positive() {
        return truePositives() + falseNegatives();
    }


    // Incrementing counters
    public void incrementTruePositives(Integer classLabel) {
        truePositives.incrementCount(classLabel, 1.0);
    }
    public void incrementTrueNegatives(Integer classLabel) {
        trueNegatives.incrementCount(classLabel, 1.0);
    }
    public void incrementFalseNegatives(Integer classLabel) {
        falseNegatives.incrementCount(classLabel, 1.0);
    }

    public void incrementFalsePositives(Integer classLabel) {
        falsePositives.incrementCount(classLabel, 1.0);
    }


    // Other misc methods
    /**
     * Adds to the confusion matrix
     * @param real the actual guess
     * @param guess the system guess
     */
    public void addToConfusion(Integer real, Integer guess) {
        confusion.add(real, guess);
    }

    /**
     * Returns the number of times the given label
     * has actually occurred
     * @param clazz the label
     * @return the number of times the label
     * actually occurred
     */
    public int classCount(Integer clazz) {
        return confusion.getActualTotal(clazz);
    }

    public double getNumRowCounter() {return (double) numRowCounter;}

    public String getClassLabel(Integer clazz) { return labelsMap.get(clazz);}
    
    /**
     * Returns the confusion matrix variable
     * @return confusion matrix variable for this evaluation
     */
    public ConfusionMatrix<Integer> getConfusionMatrix(){
        return confusion;
    }

    /** Merge the other evaluation object into this one. The result is that this Evaluation instance contains the counts
     * etc from both
     * @param other Evaluation object to merge into this one.
     */
    public void merge(Evaluation other){
        if(other == null) return;

        truePositives.incrementAll(other.truePositives);
        falsePositives.incrementAll(other.falsePositives);
        trueNegatives.incrementAll(other.trueNegatives);
        falseNegatives.incrementAll(other.falseNegatives);

        if(confusion == null){
            if(other.confusion != null) confusion = new ConfusionMatrix<>(other.confusion);
        } else {
            if (other.confusion != null) confusion.add(other.confusion);
        }
        numRowCounter += other.numRowCounter;
        if(labelsList.isEmpty()) labelsList.addAll(other.labelsList);
        if(labelsMap.isEmpty()) labelsMap.putAll(other.labelsMap);
    }
}
