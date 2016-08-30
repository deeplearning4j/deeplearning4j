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

package org.deeplearning4j.util;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * Based on the impl from:
 * https://gist.github.com/rmcgibbo/3915977
 *
 */
public class Viterbi implements Serializable {

    private double metaStability = 0.9;
    private double pCorrect = 0.99;
    private INDArray possibleLabels;
    private  int states;

    private double logPCorrect;
    private double logPIncorrect;
    private double logMetaInstability = Math.log(metaStability);
    private  double logOfDiangnalTProb;
    private double logStates;

    /**
     * The possible outcomes for the chain.
     * This should be the labels in the form of the possible outcomes (1,2,3)
     * not the binarized label matrix
     * @param possibleLabels the possible labels of the markov chain
     */

    public Viterbi(INDArray possibleLabels) {
        this.possibleLabels = possibleLabels;
        this.states = possibleLabels.length();
        this.logPCorrect = FastMath.log(pCorrect);
        this.logPIncorrect = FastMath.log(1 - pCorrect / states - 1);
        logOfDiangnalTProb = FastMath.log(1 - metaStability / states - 1);
        this.logStates = FastMath.log(states);
    }

    /**
     * Decodes the given labels, assuming its a binary label matrix
     * @param labels the labels as a binary label matrix
     * @return the decoded labels and the most likely outcome of the sequence
     */
    public Pair<Double,INDArray> decode(INDArray labels) {
        return decode(labels,true);
    }

    /**
     * Decodes a series of labels
     * @param labels the labels to decode
     * @param binaryLabelMatrix whether the label  is a binary label matrix
     * @return the most likely sequence and the sequence labels
     */
    public Pair<Double,INDArray> decode(INDArray labels,boolean binaryLabelMatrix) {
        INDArray outcomeSequence = labels.isColumnVector() || labels.isRowVector() || binaryLabelMatrix ? toOutcomesFromBinaryLabelMatrix(labels) : labels;
        int frames = outcomeSequence.length();
        INDArray V = Nd4j.ones(frames, states);
        INDArray pointers = Nd4j.zeros(frames,states);
        INDArray assigned = V.getRow(0);
        assigned.assign(logPCorrect - logStates);
        V.putRow(0,assigned);
        V.put(0,  (int) outcomeSequence.getDouble(0), logPCorrect - logStates);
        for(int t = 1; t < frames; t++) {
            for(int k = 0; k < states; k++) {
                INDArray rowLogProduct = rowOfLogTransitionMatrix(k).add(V.getRow(t  - 1));
                int maxVal = Nd4j.getBlasWrapper().iamax(rowLogProduct);
                double argMax =  rowLogProduct.max(Integer.MAX_VALUE).getDouble(0);
                V.put(t,k,argMax);
                int element = (int) outcomeSequence.getDouble(t);
                if(k == element)
                    V.put(t,k,logPCorrect + maxVal);
                else
                    V.put(t,k,logPIncorrect + maxVal);

            }
        }

        INDArray rectified = Nd4j.zeros(frames);
        rectified.put(rectified.length() - 1,V.getRow(frames - 1).max(Integer.MAX_VALUE));
        for(int t = rectified.length() - 2; t > 0; t--) {
            rectified.putScalar(t,pointers.getDouble(t + 1,(int) rectified.getDouble(t + 1)));
        }


        return new Pair<>(V.getRow(frames - 1).max(Integer.MAX_VALUE).getDouble(0),rectified);
    }

    private INDArray rowOfLogTransitionMatrix(int k) {
        INDArray row = Nd4j.ones(1,states).muli(logOfDiangnalTProb);
        row.putScalar(k,logMetaInstability);
        return row;
    }


    private INDArray toOutcomesFromBinaryLabelMatrix(INDArray outcomes) {
        INDArray ret = Nd4j.create(outcomes.rows(),1);
        for(int i = 0; i < outcomes.rows(); i++)
            ret.put(i,0, Nd4j.getBlasWrapper().iamax(outcomes.getRow(i)));
        return ret;
    }

    public double getMetaStability() {
        return metaStability;
    }

    public void setMetaStability(double metaStability) {
        this.metaStability = metaStability;
    }

    public double getpCorrect() {
        return pCorrect;
    }

    public void setpCorrect(double pCorrect) {
        this.pCorrect = pCorrect;
    }

    public INDArray getPossibleLabels() {
        return possibleLabels;
    }

    public void setPossibleLabels(INDArray possibleLabels) {
        this.possibleLabels = possibleLabels;
    }

    public int getStates() {
        return states;
    }

    public void setStates(int states) {
        this.states = states;
    }

    public double getLogPCorrect() {
        return logPCorrect;
    }

    public void setLogPCorrect(double logPCorrect) {
        this.logPCorrect = logPCorrect;
    }

    public double getLogPIncorrect() {
        return logPIncorrect;
    }

    public void setLogPIncorrect(double logPIncorrect) {
        this.logPIncorrect = logPIncorrect;
    }

    public double getLogMetaInstability() {
        return logMetaInstability;
    }

    public void setLogMetaInstability(double logMetaInstability) {
        this.logMetaInstability = logMetaInstability;
    }

    public double getLogOfDiangnalTProb() {
        return logOfDiangnalTProb;
    }

    public void setLogOfDiangnalTProb(double logOfDiangnalTProb) {
        this.logOfDiangnalTProb = logOfDiangnalTProb;
    }

    public double getLogStates() {
        return logStates;
    }

    public void setLogStates(double logStates) {
        this.logStates = logStates;
    }
}
