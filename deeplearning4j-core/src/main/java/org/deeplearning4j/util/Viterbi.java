package org.deeplearning4j.util;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.Persistable;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;

import java.io.InputStream;
import java.io.OutputStream;

/**
 * Created by agibsonccc on 5/5/14.
 */
public class Viterbi implements Persistable {

    private double metaStability = 0.9;
    private double pCorrect = 0.99;
    private DoubleMatrix possibleLabels;
    private  int states;

    private double logPCorrect = FastMath.log(pCorrect);
    private double logPIncorrect = FastMath.log(1 - pCorrect / states - 1);
    private double logMetaInstability = Math.log(metaStability);
    private  double logOfDiangnalTProb;
    private double logStates;


    public Viterbi(DoubleMatrix possibleLabels) {
        this.possibleLabels = possibleLabels;
        this.states = possibleLabels.length;
        logOfDiangnalTProb = FastMath.log(1 - metaStability / states - 1);
        this.logStates = FastMath.log(states);
    }

    /**
     * Decodes the given labels, assuming its a binary label matrix
     * @param labels the labels as a binary label matrix
     * @return the decoded labels
     */
    public Pair<Double,DoubleMatrix> decode(DoubleMatrix labels) {
        return decode(labels,true);
    }


    public Pair<Double,DoubleMatrix> decode(DoubleMatrix labels,boolean binaryLabelMatrix) {
        DoubleMatrix outcomeSequence = labels.isColumnVector() || labels.isRowVector() || binaryLabelMatrix ? toOutcomesFromBinaryLabelMatrix(labels) : labels;
        int frames = outcomeSequence.length;
        DoubleMatrix V = DoubleMatrix.ones(frames,states);
        DoubleMatrix pointers = DoubleMatrix.zeros(frames,states);
        DoubleMatrix assigned = V.getRow(0);
        MatrixUtil.assign(assigned,logPCorrect - logStates);
        V.putRow(0,assigned);
        V.put(0,(int) outcomeSequence.get(0),logPCorrect - logStates);
        for(int t = 1; t < frames; t++) {
            for(int k = 0; k < states; k++) {
                DoubleMatrix rowLogProduct = rowOfLogTransitionMatrix(k).add(V.getRow(t  - 1));
                int maxVal = SimpleBlas.iamax(rowLogProduct);
                double argMax = rowLogProduct.max();
                V.put(t,k,argMax);
                if(k == outcomeSequence.get(t))
                    V.put(t,k,logPCorrect + maxVal);
                else
                    V.put(t,k, logPIncorrect + maxVal);


            }
        }

        DoubleMatrix rectified = DoubleMatrix.zeros(frames);
        rectified.put(rectified.length - 1,V.getRow(frames - 1).max());
        for(int t = rectified.length - 2; t > 0; t--) {
            rectified.put(t,pointers.get(t + 1,(int) rectified.get(t + 1)));
        }


        return new Pair<>(V.getRow(frames - 1).max(),rectified);
    }

    private DoubleMatrix rowOfLogTransitionMatrix(int k) {
        DoubleMatrix row = DoubleMatrix.ones(1,states).mul(logOfDiangnalTProb);
        row.put(k,logMetaInstability);
        return row;
    }


    private DoubleMatrix toOutcomesFromBinaryLabelMatrix(DoubleMatrix outcomes) {
       DoubleMatrix ret = new DoubleMatrix(outcomes.rows,1);
        for(int i = 0; i < outcomes.rows; i++)
            ret.put(i,0, SimpleBlas.iamax(outcomes.getRow(i)));
        return ret;
    }


    @Override
    public void write(OutputStream os) {

    }

    @Override
    public void load(InputStream is) {

    }
}
