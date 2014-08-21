package org.deeplearning4j.util;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.nn.Persistable;


import java.io.InputStream;
import java.io.OutputStream;

/**
 * Based on the impl from:
 * https://gist.github.com/rmcgibbo/3915977
 *
 */
public class Viterbi implements Persistable {

    private double metaStability = 0.9;
    private double pCorrect = 0.99;
    private INDArray possibleLabels;
    private  int states;

    private double logPCorrect = FastMath.log(pCorrect);
    private double logPIncorrect = FastMath.log(1 - pCorrect / states - 1);
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
     * @param binaryLabelMatrix whether the label applyTransformToDestination is a binary label matrix
     * @return the most likely sequence and the sequence labels
     */
    public Pair<Double,INDArray> decode(INDArray labels,boolean binaryLabelMatrix) {
        INDArray outcomeSequence = labels.isColumnVector() || labels.isRowVector() || binaryLabelMatrix ? toOutcomesFromBinaryLabelMatrix(labels) : labels;
        int frames = outcomeSequence.length();
        INDArray V = NDArrays.ones(frames, states);
        INDArray pointers = NDArrays.zeros(frames,states);
        INDArray assigned = V.getRow(0);
        assigned.assign(logPCorrect - logStates);
        V.putRow(0,assigned);
        V.put(0, (int) outcomeSequence.getScalar(0).element(), NDArrays.scalar(logPCorrect - logStates));
        for(int t = 1; t < frames; t++) {
            for(int k = 0; k < states; k++) {
                INDArray rowLogProduct = rowOfLogTransitionMatrix(k).add(V.getRow(t  - 1));
                int maxVal = NDArrays.getBlasWrapper().iamax(rowLogProduct);
                double argMax = (double) rowLogProduct.max(Integer.MAX_VALUE).element();
                V.put(t,k,argMax);
                if(k == outcomeSequence.getScalar(t).element())
                    V.put(t,k,logPCorrect + maxVal);
                else
                    V.put(t,k, logPIncorrect + maxVal);


            }
        }

        INDArray rectified = NDArrays.zeros(frames);
        rectified.put(rectified.length() - 1,V.getRow(frames - 1).max(Integer.MAX_VALUE));
        for(int t = rectified.length() - 2; t > 0; t--) {
            rectified.put(t,pointers.getScalar(t + 1,(int) rectified.getScalar(t + 1).element()));
        }


        return new Pair<>((Double) V.getRow(frames - 1).max(Integer.MAX_VALUE).element(),rectified);
    }

    private INDArray rowOfLogTransitionMatrix(int k) {
        INDArray row = NDArrays.ones(1,states).muli(logOfDiangnalTProb);
        row.putScalar(k,logMetaInstability);
        return row;
    }


    private INDArray toOutcomesFromBinaryLabelMatrix(INDArray outcomes) {
        INDArray ret = NDArrays.create(outcomes.rows(),1);
        for(int i = 0; i < outcomes.rows(); i++)
            ret.put(i,0, NDArrays.getBlasWrapper().iamax(outcomes.getRow(i)));
        return ret;
    }


    @Override
    public void write(OutputStream os) {
        SerializationUtils.writeObject(this,os);
    }

    @Override
    public void load(InputStream is) {
        Viterbi ret = SerializationUtils.readObject(is);
        this.states = ret.states;
        this.logStates = ret.logStates;
        this.metaStability = ret.metaStability;
        this.logMetaInstability = ret.logMetaInstability;
        this.logOfDiangnalTProb = ret.logOfDiangnalTProb;
        this.logPCorrect = ret.logPCorrect;
        this.pCorrect = ret.pCorrect;

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
