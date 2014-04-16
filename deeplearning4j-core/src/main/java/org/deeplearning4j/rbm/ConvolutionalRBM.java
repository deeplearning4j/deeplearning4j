package org.deeplearning4j.rbm;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.Tensor;
import org.deeplearning4j.nn.TensorNetwork;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.util.Functions;
import org.jblas.util.Permutations;

import static org.jblas.MatrixFunctions.*;
import static org.jblas.DoubleMatrix.zeros;

public class ConvolutionalRBM extends RBM implements TensorNetwork {

    /**
     *
     */
    private static final long serialVersionUID = 6868729665328916878L;
    private int numFilters;
    private int poolRows;
    private int poolColumns;
    //top down signal from hidden feature maps to visibles
    private Tensor visI;
    //bottom up signal from visibles to hiddens
    private Tensor hidI;
    //visible unit expectation
    private DoubleMatrix visExpectation;
    //hidden unit expectation
    private DoubleMatrix hiddenExpectation;
    //initial hidden expectation for gradients
    private DoubleMatrix initialHiddenExpectation;
    private DoubleMatrix fmSize;
    private Tensor W;
    private DoubleMatrix poolingLayer;
    private int[] stride;


    protected ConvolutionalRBM() {}




    protected ConvolutionalRBM(DoubleMatrix input, int nVisible, int n_hidden, DoubleMatrix W,
                               DoubleMatrix hbias, DoubleMatrix vBias, RandomGenerator rng,double fanIn,RealDistribution dist) {
        super(input, nVisible, n_hidden, W, hbias, vBias, rng,fanIn,dist);
    }


    public DoubleMatrix visibleExpectation(DoubleMatrix visible,double bias) {
        DoubleMatrix filterMatrix = new DoubleMatrix(numFilters);
        for(int k = 0; k < numFilters; k++) {
            DoubleMatrix next = MatrixUtil.convolution2D(visible,
                    visible.columns,
                    visible.rows, this.getW().getRow(k), this.getW().rows, this.getW().columns).add(this.getvBias().add(bias)).transpose();
            filterMatrix.putRow(k,next);
        }

        //   filterMatrix = pool(filterMatrix);

        filterMatrix.addi(1);
        filterMatrix = MatrixUtil.oneDiv(filterMatrix);

        //replace with actual function later, sigmoid is only one possible option
        return MatrixUtil.sigmoid(filterMatrix);

    }


    private void init() {
        W = new Tensor(getnVisible(),getnHidden(),numFilters);
        visI = Tensor.zeros(getnVisible(),getnHidden(),numFilters);
        hidI = Tensor.zeros(getnVisible() - numFilters + 1,getnHidden() - numFilters + 1,numFilters);
    }



    public DoubleMatrix pooledExpectation(DoubleMatrix visible,double bias) {
        DoubleMatrix filterMatrix = new DoubleMatrix(numFilters);
        for(int k = 0; k < numFilters; k++) {
            DoubleMatrix next = MatrixUtil.convolution2D(visible,
                    visible.columns,
                    visible.rows, this.getW().getRow(k), this.getW().rows, this.getW().columns).add(this.gethBias().add(bias)).transpose();
            filterMatrix.putRow(k,next);
        }

       // filterMatrix = pool(filterMatrix);

        filterMatrix.addi(1);
        filterMatrix = MatrixUtil.oneDiv(filterMatrix);

        return filterMatrix;

    }

    public Tensor pooledActivations(Tensor input) {
        int nCols = input.columns();
        int rows = input.rows();
        Tensor ret = Tensor.zeros(rows,nCols,input.slices());
        for(int i = 0;i < Math.ceil(nCols / stride[0]); i++) {
            int rowsMin = (i - 1) * stride[0] + 1;
            int rowsMax = i * stride[0];
            for(int j = 0; j < Math.ceil(nCols / stride[1]); j++) {
                int cols = (j - 1) * stride[1] + 1;
                int colsMax = j * stride[1];
                double blockVal = input.get(new int[] {
                        rowsMin,rowsMax
                },new int[] {
                        cols,colsMax
                }).columnsSums().sum();



            }

        }
       return ret;
    }


    /**
     * Binomial sampling of the hidden values given visible
     *
     * @param v the visible values
     * @return a binomial distribution containing the expected values and the samples
     */
    @Override
    public Pair<DoubleMatrix, DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v) {
        return super.sampleHiddenGivenVisible(v);
    }

    @Override
    public NeuralNetworkGradient getGradient(Object[] params) {
        return super.getGradient(params);
    }

    /**
     * Guess the visible values given the hidden
     *
     * @param h
     * @return
     */
    @Override
    public Pair<DoubleMatrix, DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h) {
        for(int i = 0;i < numFilters; i++) {

        }

        return super.sampleVisibleGivenHidden(h);
    }

    @Override
    public Tensor getWTensor() {
        return null;
    }

    @Override
    public void setW(Tensor w) {

    }
}
