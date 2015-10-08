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
 *  *    WITHOUInteger WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Evaluation metrics: precision, recall, f1
 * @author Adam Gibson
 *
 */
public class Evaluation<T extends Comparable<? super T>> implements Serializable {

    private Counter<Integer> truePositives = new Counter<>();
    private Counter<Integer> falsePositives = new Counter<>();
    private Counter<Integer> trueNegatives = new Counter<>();
    private Counter<Integer> falseNegatives = new Counter<>();
    private ConfusionMatrix<Integer> confusion;
    private int numRowCounter = 0;
    private List<Integer> labelsList = new ArrayList<>();
    private Map<Integer, String> labelsMap = new HashMap<>();
    private static Logger log = LoggerFactory.getLogger(Evaluation.class);

    // Empty constructor
    public Evaluation() {}

    // Constructor that takes number of output classes
    public Evaluation(int numClasses) {
        for(int i = 0; i < numClasses; i++)
            labelsList.add(i);
        confusion = new ConfusionMatrix<>(labelsList);
    }

    public Evaluation(List<String> labels) {
        int i = 0;
        for (String label : labels){
            this.labelsMap.put(i, label);
            i++;
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
     * @param realOutcomes the real outcomes (usually binary)
     * @param guesses the guesses (usually a probability vector)
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

    // Method to print the classification report
    public String stats() {
        StringBuilder builder = new StringBuilder().append("\n");
        List<Integer> classes = confusion.getClasses();

        if (labelsMap.isEmpty()){
            for (Integer clazz : classes) {
                for (Integer clazz2 : classes) {
                    int count = confusion.getCount(clazz, clazz2);
                    if (count != 0)
                        builder.append("\n Examples labeled as " + clazz + " classified by model as " + clazz2 + ": " + count + " times\n");
                }
            }
        } else {
            for (Integer clazz : classes) {
                for (Integer clazz2 : classes) {
                    int count = confusion.getCount(clazz, clazz2);
                    if (count != 0)
                        builder.append("\n Examples labeled as "+ labelsMap.get(clazz) + " classified by model as " + labelsMap.get(clazz2) + ": " + count + " times\n");
                }
            }
        }

        DecimalFormat df = new DecimalFormat("#.####");
        builder.append("\n==========================Scores========================================");
        builder.append("\n Accuracy:  " + df.format(accuracy()));
        builder.append("\n Precision: " + df.format(precision()));
        builder.append("\n Recall:    " + df.format(recall()));
        builder.append("\n F1 Score:  " + f1());
        builder.append("\n===========================================================================");
        return builder.toString();
    }


    /**
     * Returns the precision for a given label
     * @param classLabel the label
     * @return the precision for the label
     */
    public double precision(Integer classLabel) {
        double tpCount = truePositives.getCount(classLabel);
        double fpCount = falsePositives.getCount(classLabel);
        if (tpCount == 0)
            return 0;
        return tpCount / (tpCount + fpCount);
    }

    /**
     * Total precision based on guesses so far
     * @return the total precision based on guesses so far
     *
     */
    public double precision() {
        double precisionAcc = 0.0;
        double classCount = 0.0;
        for(Integer classLabel : confusion.getClasses()) {
            precisionAcc += precision(classLabel);
            if (truePositives.getCount(classLabel) > 0) {
               classCount += 1.0;
            }
        }
        return precisionAcc / classCount;
    }

    /**
     * Get the recall for a particular class label
     * @param classLabel Integer the indicate which class
     * @return Recall rate as a double
     */
    public double recall(Integer classLabel) {
        double tpCount = truePositives.getCount(classLabel);
        double fnCount = falseNegatives.getCount(classLabel);

        if (tpCount == 0)
            return 0;

        return tpCount / (tpCount + fnCount);
    }

    /**
     * Returns the recall for the outcomes
     * @return the recall for the outcomes
     */
    public double recall() {
        double recallAcc = 0.0;
        double classCount = 0.0;
        for(Integer classLabel : confusion.getClasses()) {
            recallAcc += recall(classLabel);
            if (truePositives.getCount(classLabel) > 0) {
                classCount += 1.0;
            }

        }
        return recallAcc / classCount;
    }

    /**
     * Calculate f1 score for a given class
     * @param classLabel the label to calculate f1 for
     * @return the f1 score for the given label
     */
    public double f1(Integer classLabel) {
        double precision = precision(classLabel);
        double recall = recall();
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
        truePositives.incrementCount(classLabel, 1.0);
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


}
