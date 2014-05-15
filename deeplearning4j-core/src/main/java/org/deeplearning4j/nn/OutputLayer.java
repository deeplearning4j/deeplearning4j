package org.deeplearning4j.nn;

import static org.deeplearning4j.util.MatrixUtil.log;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.jblas.MatrixFunctions.pow;
import static org.jblas.MatrixFunctions.sqrt;

import java.io.Serializable;

import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.nn.gradient.OutputLayerGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.optimize.OutputLayerOptimizer;
import org.deeplearning4j.optimize.TrainingEvaluator;
import org.deeplearning4j.optimize.VectorizedDeepLearningGradientAscent;
import org.deeplearning4j.optimize.VectorizedNonZeroStoppingConjugateGradient;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;

/**
 * Logistic regression implementation with jblas.
 * @author Adam Gibson
 *
 */
public class OutputLayer implements Serializable {

    private static final long serialVersionUID = -7065564817460914364L;
    //number of inputs from final hidden layer
    private int nIn;
    //number of outputs for labeling
    private int nOut;
    //current input and label matrices
    private DoubleMatrix input,labels;
    //weight matrix
    private DoubleMatrix W;
    //bias
    private DoubleMatrix b;
    //weight decay; l2 regularization
    private double l2 = 0.01;
    private boolean useRegularization = false;
    private boolean useAdaGrad = true;
    private AdaGrad adaGrad,biasAdaGrad;
    private boolean normalizeByInputRows = true;
    private OptimizationAlgorithm optimizationAlgorithm;
    private LossFunction lossFunction;
    private ActivationFunction activationFunction;



    private OutputLayer() {}

    /**
     * MSE: Mean Squared Error: Linear Regression
     * EXPLL: Exponential log likelihood: Poisson Regression
     * XENT: Cross Entropy: Binary Classification
     * SOFTMAX: Softmax Regression
     * RMSE_XENT: RMSE Cross Entropy
     *
     *
     */
    public static enum LossFunction {
        MSE,EXPLL,XENT,MCXENT,RMSE_XENT
    }



    public OutputLayer(DoubleMatrix input, DoubleMatrix labels, int nIn, int nOut) {
        this.input = input;
        this.labels = labels;
        this.nIn = nIn;
        this.nOut = nOut;
        W = DoubleMatrix.zeros(nIn,nOut);
        adaGrad = new AdaGrad(nIn,nOut);
        b = DoubleMatrix.zeros(1,nOut);
        biasAdaGrad = new AdaGrad(b.rows,b.columns);
    }

    public OutputLayer(DoubleMatrix input, int nIn, int nOut) {
        this(input,null,nIn,nOut);
    }

    public OutputLayer(int nIn, int nOut) {
        this(null,null,nIn,nOut);
    }

    /**
     * Train with current input and labels
     * with the given learning rate
     * @param lr the learning rate to use
     */
    public  void train(double lr) {
        train(input,labels,lr);
    }



    /**
     * Train with the given input
     * and the currently set labels
     * @param x the input to use
     * @param lr the learning rate to use
     */
    public  void train(DoubleMatrix x,double lr) {
        adaGrad.setMasterStepSize(lr);
        biasAdaGrad.setMasterStepSize(lr);

        MatrixUtil.complainAboutMissMatchedMatrices(x, labels);

        train(x,labels,lr);

    }

    /**
     * Run conjugate gradient with the given x and y
     * @param x the input to use
     * @param y the labels to use
     * @param learningRate
     * @param epochs
     */
    public  void trainTillConvergence(DoubleMatrix x,DoubleMatrix y, double learningRate,int epochs) {
        MatrixUtil.complainAboutMissMatchedMatrices(x, y);
        adaGrad.setMasterStepSize(learningRate);
        biasAdaGrad.setMasterStepSize(learningRate);

        this.input = x;
        this.labels = y;
        trainTillConvergence(learningRate,epochs);

    }


    /**
     * Run the optimization algorithm for training
     * @param learningRate the learning rate to train with
     * @param numEpochs the number of epochs
     * @param eval the training evaluator to use for early stopping (where applicable)
     */
    public  void trainTillConvergence(DoubleMatrix labels,double learningRate, int numEpochs,TrainingEvaluator eval) {

        this.labels = labels;
        OutputLayerOptimizer opt = new OutputLayerOptimizer(this, learningRate);
        adaGrad.setMasterStepSize(learningRate);
        biasAdaGrad.setMasterStepSize(learningRate);

        if(optimizationAlgorithm == OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(opt);
            g.setTolerance(1e-3);
            g.setTrainingEvaluator(eval);
            g.setMaxIterations(numEpochs);
            g.optimize(numEpochs);

        }

        else {
            VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(opt);
            g.setTolerance(1e-3);
            g.setTrainingEvaluator(eval);
            g.optimize(numEpochs);

        }


    }

    /**
     * Run the optimization algorithm for training
     * @param learningRate the learning rate to train with
     * @param numEpochs the number of epochs
     * @param eval the training evaluator to use for early stopping (where applicable)
     */
    public  void trainTillConvergence(double learningRate, int numEpochs,TrainingEvaluator eval) {
        OutputLayerOptimizer opt = new OutputLayerOptimizer(this, learningRate);
        adaGrad.setMasterStepSize(learningRate);
        biasAdaGrad.setMasterStepSize(learningRate);

        if(optimizationAlgorithm == OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(opt);
            g.setTolerance(1e-3);
            g.setTrainingEvaluator(eval);
            g.setMaxIterations(numEpochs);
            g.optimize(numEpochs);

        }

        else {
            VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(opt);
            g.setTolerance(1e-3);
            g.setTrainingEvaluator(eval);
            g.optimize(numEpochs);

        }


    }


    /**
     * Run conjugate gradient
     * @param learningRate the learning rate to train with
     * @param numEpochs the number of epochs
     */
    public  void trainTillConvergence(double learningRate, int numEpochs) {
        trainTillConvergence(learningRate,numEpochs,null);
    }


    /**
     * Averages the given logistic regression
     * from a mini batch in to this one
     * @param l the logistic regression to average in to this one
     * @param batchSize  the batch size
     */
    public void merge(OutputLayer l,int batchSize) {
        if(useRegularization) {

            W.addi(l.W.subi(W).div(batchSize));
            b.addi(l.b.subi(b).div(batchSize));
        }

        else {
            W.addi(l.W.subi(W));
            b.addi(l.b.subi(b));
        }

    }

    /**
     * Objective function:  the specified objective
     * @return the score for the objective
     */
    public  double score() {
        MatrixUtil.complainAboutMissMatchedMatrices(input, labels);
        DoubleMatrix z = output(input);
        switch (lossFunction) {
            case MCXENT:
                DoubleMatrix mcXEntLogZ = log(z);
                return - labels.mul(mcXEntLogZ).columnSums().sum() / labels.rows;
            case XENT:
                DoubleMatrix xEntLogZ = log(z);
                DoubleMatrix xEntOneMinusLabelsOut = oneMinus(labels);
                DoubleMatrix xEntOneMinusLogOneMinusZ = oneMinus(log(z));
                return -labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).mul(xEntOneMinusLogOneMinusZ).columnSums().sum() / labels.rows;
            case RMSE_XENT:
                return sqrt(pow(labels.sub(z),2)).columnSums().sum() / labels.rows;
            case MSE:
                DoubleMatrix mseDelta = labels.sub(z);
                return 0.5 * pow(mseDelta, 2).columnSums().sum() / labels.rows;
            case EXPLL:
                DoubleMatrix expLLLogZ = log(z);
                return -z.sub(labels.mul(expLLLogZ)).columnSums().sum() / labels.rows;


        }

        throw new IllegalStateException("No loss function found");




    }

    /**
     * Train on the given inputs and labels.
     * This will assign the passed in values
     * as fields to this logistic function for
     * caching.
     * @param x the inputs to train on
     * @param y the labels to train on
     * @param lr the learning rate
     */
    public  void train(DoubleMatrix x,DoubleMatrix y, double lr) {
        adaGrad.setMasterStepSize(lr);
        biasAdaGrad.setMasterStepSize(lr);


        MatrixUtil.complainAboutMissMatchedMatrices(x, y);

        this.input = x;
        this.labels = y;

        //DoubleMatrix regularized = W.transpose().mul(l2);
        OutputLayerGradient gradient = getGradient(lr);



        W.addi(gradient.getwGradient());
        b.addi(gradient.getbGradient());

    }





    @Override
    protected OutputLayer clone()  {
        OutputLayer reg = new OutputLayer();
        reg.b = b.dup();
        reg.W = W.dup();
        reg.l2 = this.l2;
        if(this.labels != null)
            reg.labels = this.labels.dup();
        reg.nIn = this.nIn;
        reg.nOut = this.nOut;
        reg.useRegularization = this.useRegularization;
        reg.normalizeByInputRows = this.normalizeByInputRows;
        reg.biasAdaGrad = this.biasAdaGrad;
        reg.adaGrad = this.adaGrad;
        reg.useAdaGrad = this.useAdaGrad;
        reg.setOptimizationAlgorithm(this.getOptimizationAlgorithm());
        reg.lossFunction = this.lossFunction;
        if(this.input != null)
            reg.input = this.input.dup();
        return reg;
    }

    /**
     * Gets the gradient from one training iteration
     * @param lr the learning rate to use for training
     * @return the gradient (bias and weight matrix)
     */
    public OutputLayerGradient getGradient(double lr) {
        MatrixUtil.complainAboutMissMatchedMatrices(input, labels);


        if(adaGrad.historicalGradient.rows != W.rows || adaGrad.historicalGradient.columns != W.columns)
            adaGrad = new AdaGrad(W.rows,W.columns);
        if(biasAdaGrad.historicalGradient.rows != b.rows || biasAdaGrad.historicalGradient.columns != b.columns)
            biasAdaGrad = new AdaGrad(b.rows,b.columns);
        adaGrad.setMasterStepSize(lr);
        biasAdaGrad.setMasterStepSize(lr);


        //input activation
        DoubleMatrix netOut = output(input);
        //difference of outputs
        DoubleMatrix dy = labels.sub(netOut);
        //weight decay
        if(normalizeByInputRows)
            dy.divi(input.rows);

        DoubleMatrix wGradient = getWeightGradient();
        if(useAdaGrad)
            wGradient.muli(adaGrad.getLearningRates(wGradient));
        else
            wGradient.muli(lr);

        if(useAdaGrad)
            dy.muliRowVector(biasAdaGrad.getLearningRates(dy.columnMeans()));
        else
            dy.muli(lr);

        if(normalizeByInputRows)
            dy.divi(input.rows);


        DoubleMatrix bGradient = dy;
        return new OutputLayerGradient(wGradient,bGradient);


    }


    private DoubleMatrix getWeightGradient() {
        DoubleMatrix z = output(input);
        switch (lossFunction) {
            case MCXENT:
                //input activation
                DoubleMatrix p_y_given_x = sigmoid(input.mmul(W).addRowVector(b));
                //difference of outputs
                DoubleMatrix dy = labels.sub(p_y_given_x);
                return input.transpose().mmul(dy);

            case XENT:
                DoubleMatrix xEntDiff = z.sub(labels);
                return input.transpose().mmul(xEntDiff.div(z.mul(oneMinus(z))));
            case MSE:
                DoubleMatrix mseDelta = labels.sub(z);
                return input.transpose().mmul(mseDelta.neg());
            case EXPLL:
                return input.transpose().mmul(oneMinus(labels).div(z));


        }

        throw new IllegalStateException("Invalid loss function");

    }



    /**
     * Classify input
     * @param x the input (can either be a matrix or vector)
     * If it's a matrix, each row is considered an example
     * and associated rows are classified accordingly.
     * Each row will be the likelihood of a label given that example
     * @return a probability distribution for each row
     */
    public  DoubleMatrix output(DoubleMatrix x) {
        if(x == null)
            throw new IllegalArgumentException("No null input allowed");
        this.input = x;
        if(W == null)
            throw new IllegalStateException("EH?");
        if(b == null)
            throw new IllegalStateException("EH?");
        try {
            return activationFunction.apply(this.input.mmul(W).addRowVector(b));

        }catch(Exception e) {
            throw new IllegalStateException("EH?");
        }

    }



    public  int getnIn() {
        return nIn;
    }

    public  void setnIn(int nIn) {
        this.nIn = nIn;
    }

    public  int getnOut() {
        return nOut;
    }

    public  void setnOut(int nOut) {
        this.nOut = nOut;
    }

    public  DoubleMatrix getInput() {
        return input;
    }

    public  void setInput(DoubleMatrix input) {
        this.input = input;
    }

    public  DoubleMatrix getLabels() {
        return labels;
    }

    public  void setLabels(DoubleMatrix labels) {
        this.labels = labels;
    }

    public  DoubleMatrix getW() {
        return W;
    }

    public  void setW(DoubleMatrix w) {
        W = w;
    }

    public  DoubleMatrix getB() {
        return b;
    }

    public  void setB(DoubleMatrix b) {
        this.b = b;
    }

    public  double getL2() {
        return l2;
    }

    public  void setL2(double l2) {
        this.l2 = l2;
    }

    public  boolean isUseRegularization() {
        return useRegularization;
    }

    public  void setUseRegularization(boolean useRegularization) {
        this.useRegularization = useRegularization;
    }

    public AdaGrad getBiasAdaGrad() {
        return biasAdaGrad;
    }


    public AdaGrad getAdaGrad() {
        return adaGrad;
    }



    public synchronized boolean isNormalizeByInputRows() {
        return normalizeByInputRows;
    }



    public synchronized void setNormalizeByInputRows(boolean normalizeByInputRows) {
        this.normalizeByInputRows = normalizeByInputRows;
    }



    public boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public void setAdaGrad(AdaGrad adaGrad) {
        this.adaGrad = adaGrad;
    }

    public void setBiasAdaGrad(AdaGrad biasAdaGrad) {
        this.biasAdaGrad = biasAdaGrad;
    }

    public OptimizationAlgorithm getOptimizationAlgorithm() {
        return optimizationAlgorithm;
    }



    public void setOptimizationAlgorithm(OptimizationAlgorithm optimizationAlgorithm) {
        this.optimizationAlgorithm = optimizationAlgorithm;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public static class Builder {
        private DoubleMatrix W;
        private OutputLayer ret;
        private DoubleMatrix b;
        private double l2;
        private int nIn;
        private int nOut;
        private DoubleMatrix input;
        private boolean useRegualarization;
        private ActivationFunction activationFunction = Activations.softmax();
        private boolean useAdaGrad = true;
        private boolean normalizeByInputRows = true;
        private OptimizationAlgorithm optimizationAlgorithm;
        private LossFunction lossFunction = LossFunction.MCXENT;


        public Builder withActivationFunction(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder withLossFunction(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        public Builder optimizeBy(OptimizationAlgorithm optimizationAlgorithm) {
            this.optimizationAlgorithm = optimizationAlgorithm;
            return this;
        }

        public Builder normalizeByInputRows(boolean normalizeByInputRows) {
            this.normalizeByInputRows = normalizeByInputRows;
            return this;
        }

        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }


        public Builder withL2(double l2) {
            this.l2 = l2;
            return this;
        }

        public Builder useRegularization(boolean regularize) {
            this.useRegualarization = regularize;
            return this;
        }

        public Builder withWeights(DoubleMatrix W) {
            this.W = W;
            return this;
        }

        public Builder withBias(DoubleMatrix b) {
            this.b = b;
            return this;
        }

        public Builder numberOfInputs(int nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder numberOfOutputs(int nOut) {
            this.nOut = nOut;
            return this;
        }

        public OutputLayer build() {
            ret = new OutputLayer(input, nIn, nOut);
            if(W != null)
                ret.W = W;
            if(b != null)
                ret.b = b;
            ret.optimizationAlgorithm = optimizationAlgorithm;
            ret.normalizeByInputRows = normalizeByInputRows;
            ret.useRegularization = useRegualarization;
            ret.l2 = l2;
            ret.useAdaGrad = useAdaGrad;
            ret.lossFunction = lossFunction;
            ret.activationFunction = activationFunction;

            return ret;
        }

    }

}
