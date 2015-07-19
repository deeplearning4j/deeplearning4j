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

package org.deeplearning4j.models.rntn;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.eval.ConfusionMatrix;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.DecimalFormat;
import java.util.List;
import java.util.Set;

/**
 * Recursive counter for an RNTN
 *
 * @author Adam Gibson
 */
public class RNTNEval {

    private Counter<Integer> truePositives = new Counter<>();
    private Counter<Integer> falsePositives = new Counter<>();
    private Counter<Integer> falseNegatives = new Counter<>();
    private ConfusionMatrix<Integer> confusionMatrix = new ConfusionMatrix<>();
    private static final Logger log = LoggerFactory.getLogger(RNTNEval.class);


    /**
     * Eval the RNTN
     * @param rntn
     * @param trees
     */

    public void eval(RNTN rntn, List<Tree> trees) {
        for(Tree t : trees) {
            rntn.forwardPropagateTree(t);
            count(t);
        }

    }

    private void count(Tree tree) {
        if(tree.isLeaf())
            return;
        if(tree.prediction() == null) {
            return;
        }

        for(Tree t : tree.children())
            count(t);
        int treeGoldLabel = tree.goldLabel();
        int predictionLabel = Nd4j.getBlasWrapper().iamax(tree.prediction());
        confusionMatrix.add(treeGoldLabel, predictionLabel);
    }


    public void incrementTruePositives(int clazz, int i){
        truePositives.incrementCount(clazz, i);
    }

    public void incrementFalsePositives(int clazz, int i){
        falsePositives.incrementCount(clazz, i);
    }

    public void incrementFalseNegatives(int clazz, int i){
        falseNegatives.incrementCount(clazz, i);
    }


    /**
     * Print the summary of the rntnresults
     * @return the summary of the rntn
     */
    public String stats() {
        StringBuilder builder = new StringBuilder()
                .append("\n");
        List<Integer> classes = confusionMatrix.getClasses();
        for(Integer clazz : classes) {
            for(Integer clazz2 : classes) {
                int count = confusionMatrix.getCount(clazz, clazz2);
                if(count != 0)
                    builder.append("\nActual Class " + clazz + " was predicted with Predicted " + clazz2 + " with count " + count  + " times\n");
                if (clazz == clazz2) {
                    incrementTruePositives(clazz, count);
                } else {
                    incrementFalsePositives(clazz2, count);
                }
            }
            int falseNegatives = confusionMatrix.getActualTotal(clazz) - confusionMatrix.getPredictedTotal(clazz);
            if (falseNegatives > 0) incrementFalseNegatives(clazz, falseNegatives);
        }
        DecimalFormat df = new DecimalFormat("#.####");
        builder.append("\n==========================Scores========================================");
        builder.append("\n Precision: " + df.format(precision()));
        builder.append("\n Recall: " + df.format(recall()));
        builder.append("\n F1 Score: " + df.format(f1()));
        builder.append("\n===========================================================================");
        return builder.toString();
    }


    /**
     * Total precision based on guesses so far
     * @return the total precision based on guesses so far
     *
     */
    public double precision() {
        double prec = 0.0;
        for(Integer i : confusionMatrix.getClasses()) {
            prec += precision(i);
        }
        return prec / (double) confusionMatrix.getClasses().size();
    }

    /**
     * Returns the precision for a given label
     * @param i the label
     * @return the precision for the label
     */
    public double precision(int i) {
        if(truePositives.getCount(i) == 0)
            return 0;
        return truePositives.getCount(i) / (truePositives.getCount(i) + falsePositives.getCount(i));
    }

    /**
     * Returns the recall for the outcomes
     * @return the recall for the outcomes
     */
    public double recall() {
        double r = 0.0;
        for(Integer i : confusionMatrix.getClasses()) {
            r += recall(i);
        }
        return r / (double) confusionMatrix.getClasses().size();
    }

    public double recall(int i) {
        if(truePositives.getCount(i) == 0)
            return 0;
        return truePositives.getCount(i) / (truePositives.getCount(i) + falseNegatives.getCount(i));
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
     * Calculate f1 score for a given class
     * @param i the label to calculate f1 for
     * @return the f1 score for the given label
     */
    public double f1(int i) {
        double precision = precision(i);
        double recall = recall();
        if(precision == 0 || recall == 0)
            return 0;
        return 2.0 * ((precision * recall / (precision + recall)));
    }


}
