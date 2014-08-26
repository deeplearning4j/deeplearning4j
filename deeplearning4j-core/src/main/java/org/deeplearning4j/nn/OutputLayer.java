package org.deeplearning4j.nn;

import static org.deeplearning4j.linalg.ops.transforms.Transforms.*;

import java.io.Serializable;

import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.util.LinAlgExceptions;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;

import org.deeplearning4j.nn.gradient.OutputLayerGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.optimize.*;


/**
 * Output layer with different objective functions for different objectives.
 * This includes classification as well as prediction
 * @author Adam Gibson
 *
 */
public class OutputLayer implements Serializable,Output {

    private static final long serialVersionUID = -7065564817460914364L;
    //number of inputs from final hidden layer
    private int nIn;
    //number of outputs for labeling
    private int nOut;
    //current input and label matrices
    private INDArray input,labels;
    //weight matrix
    private INDArray W;
    //bias
    private INDArray b;
    //weight decay; l2 regularization
    private double l2 = 0.01;
    private boolean useRegularization = false;
    private boolean useAdaGrad = true;
    private AdaGrad adaGrad,biasAdaGrad;
    private boolean normalizeByInputRows = true;
    private OptimizationAlgorithm optimizationAlgorithm;
    private LossFunction lossFunction;
    private ActivationFunction activationFunction;
    private double dropOut;
    private INDArray dropoutMask;
    private boolean concatBiases = false;
    private boolean constrainGradientToUniNorm = false;
    private WeightInit weightinit;

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
        MSE,EXPLL,XENT,MCXENT,RMSE_XENT,SQUARED_LOSS
    }

    public OutputLayer(INDArray input, INDArray labels, int nIn, int nOut) {
        this(input,labels,nIn,nOut,null);
    }

    public OutputLayer(INDArray input, INDArray labels, int nIn, int nOut,WeightInit weightInit) {
        this.input = input;
        this.labels = labels;
        this.nIn = nIn;
        this.nOut = nOut;
        this.weightinit = weightInit;
        if(weightInit != null) {
            W = WeightInitUtil.initWeights(nIn,nOut,weightInit,activationFunction);
        }
        else
            W = NDArrays.zeros(nIn,nOut);
        adaGrad = new AdaGrad(nIn,nOut);
        b = NDArrays.zeros(1,nOut);
        biasAdaGrad = new AdaGrad(b.rows(),b.columns());
    }

    public OutputLayer(INDArray input, int nIn, int nOut) {
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
     * and the currently applyTransformToDestination labels
     * @param x the input to use
     * @param lr the learning rate to use
     */
    public  void train(INDArray x,double lr) {
        adaGrad.setMasterStepSize(lr);
        biasAdaGrad.setMasterStepSize(lr);

        LinAlgExceptions.assertRows(x,labels);

        train(x,labels,lr);

    }

    /**
     * Run conjugate gradient with the given x and y
     * @param x the input to use
     * @param y the labels to use
     * @param learningRate
     * @param epochs
     */
    public  void trainTillConvergence(INDArray x,INDArray y, double learningRate,int epochs) {
        LinAlgExceptions.assertRows(x,y);
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
    public  void trainTillConvergence(INDArray labels,double learningRate, int numEpochs,TrainingEvaluator eval) {

        this.labels = labels;
        OutputLayerOptimizer opt = new OutputLayerOptimizer(this, learningRate);
        adaGrad.setMasterStepSize(learningRate);
        biasAdaGrad.setMasterStepSize(learningRate);

        if(optimizationAlgorithm == OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(opt);
            g.setTolerance(1e-3f);
            g.setTrainingEvaluator(eval);
            g.setMaxIterations(numEpochs);
            g.optimize(numEpochs);

        }
        else if(optimizationAlgorithm == OptimizationAlgorithm.HESSIAN_FREE) {
            StochasticHessianFree o = new StochasticHessianFree(opt,null);
            o.setTolerance(1e-3f);
            o.setTrainingEvaluator(eval);
            o.optimize(numEpochs);
        }

        else {
            VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(opt);
            g.setTolerance(1e-3f);
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
            g.setTolerance(1e-3f);
            g.setTrainingEvaluator(eval);
            g.setMaxIterations(numEpochs);
            g.optimize(numEpochs);

        }

        else {
            VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(opt);
            g.setTolerance(1e-3f);
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
    public  float score() {
        LinAlgExceptions.assertRows(input,labels);
        INDArray z = output(input);
        float ret = 0;
        double reg = 0.5 * l2;


        switch (lossFunction) {
            case MCXENT:
                INDArray mcXEntLogZ = log(z);
                ret = -(float) labels.mul(mcXEntLogZ).sum(1).sum(Integer.MAX_VALUE).element() / labels.rows();
                break;
            case XENT:
                INDArray xEntLogZ = log(z);
                INDArray xEntOneMinusLabelsOut = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ = log(z).rsub(1);
                ret = -(float) labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).mul(xEntOneMinusLogOneMinusZ).sum(1).sum(Integer.MAX_VALUE).element() / labels.rows();
                break;
            case RMSE_XENT:
                ret = (float) pow(labels.sub(z),2).sum(1).sum(Integer.MAX_VALUE).element() / labels.rows();
                break;
            case MSE:
                INDArray mseDelta = labels.sub(z);
                ret =  0.5f * (float) (pow(mseDelta, 2).sum(1).sum(Integer.MAX_VALUE)).element() / labels.rows();
                break;
            case EXPLL:
                INDArray expLLLogZ = log(z);
                ret = -(float) z.sub(labels.mul(expLLLogZ)).sum(1).sum(Integer.MAX_VALUE).element() / labels.rows();
                break;
            case SQUARED_LOSS:
                ret = (float) pow(labels.sub(z),2).sum(1).sum(Integer.MAX_VALUE).element() / labels.rows();


        }

        if(useRegularization && l2 > 0)
            ret += reg;


        return ret;


    }


    protected void applyDropOutIfNecessary(INDArray input) {
        if(dropOut > 0) {
            this.dropoutMask = NDArrays.rand(input.rows(), this.nOut).gt(dropOut);
        }

        else
            this.dropoutMask = NDArrays.ones(input.rows(),this.nOut);

        //actually apply drop out
        input.muli(dropoutMask);

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
    public  void train(INDArray x,INDArray y, double lr) {

        adaGrad.setMasterStepSize(lr);
        biasAdaGrad.setMasterStepSize(lr);


        LinAlgExceptions.assertRows(input,labels);

        this.input = x;
        this.labels = y;

        //INDArray regularized = W.transpose().mul(l2);
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
        reg.constrainGradientToUniNorm = this.constrainGradientToUniNorm;
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
        reg.concatBiases = this.concatBiases;
        reg.weightinit = weightinit;
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
        LinAlgExceptions.assertRows(input,labels);

        adaGrad.setMasterStepSize(lr);
        biasAdaGrad.setMasterStepSize(lr);


        //input activation
        INDArray netOut = output(input);
        //difference of outputs
        INDArray dy = labels.sub(netOut);
        //weight decay
        if(normalizeByInputRows)
            dy.divi(input.rows());

        INDArray wGradient = getWeightGradient();
        if(useAdaGrad)
            wGradient.muli(adaGrad.getLearningRates(wGradient));
        else
            wGradient.muli(lr);

        if(useAdaGrad)
            dy.muliRowVector(biasAdaGrad.getLearningRates(dy.mean(1)));
        else
            dy.muli(lr);

        if(normalizeByInputRows)
            dy.divi(input.rows());


        INDArray bGradient = dy;
        if(constrainGradientToUniNorm) {
            wGradient.divi(wGradient.norm2(Integer.MAX_VALUE));
            bGradient.divi(bGradient.norm2(Integer.MAX_VALUE));
        }


        return new OutputLayerGradient(wGradient,bGradient);


    }


    private INDArray getWeightGradient() {
        INDArray z = output(input);

        switch (lossFunction) {
            case MCXENT:
                //input activation
                INDArray p_y_given_x = sigmoid(input.mmul(W).addRowVector(b));
                //difference of outputs
                INDArray dy = labels.sub(p_y_given_x);
                return input.transpose().mmul(dy);

            case XENT:
                INDArray xEntDiff = z.sub(labels);
                return input.transpose().mmul(xEntDiff.div(z.mul(z.rsub(1))));
            case MSE:
                INDArray mseDelta = labels.sub(z);
                return input.transpose().mmul(mseDelta.neg());
            case EXPLL:
                return input.transpose().mmul(labels.rsub(1).divi(z));
            case RMSE_XENT:
                return input.transpose().mmul(pow(labels.sub(z),2));
            case SQUARED_LOSS:
                return input.transpose().mmul(pow(labels.sub(z),2));



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
    public  INDArray preOutput(INDArray x) {
        if(x == null)
            throw new IllegalArgumentException("No null input allowed");

        this.input = x;

        INDArray ret = this.input.mmul(W);
        if(concatBiases)
            ret = NDArrays.concatHorizontally(ret,b);
        else
            ret.addiRowVector(b);
        return ret;


    }

    /**
     * Returns the predictions for each example in the dataset
     * @param d the matrix to predict
     * @return the prediction for the dataset
     */
    public int[] predict(INDArray d) {
        INDArray output = output(d);
        int[] ret = new int[d.rows()];
        for(int i = 0; i < ret.length; i++)
            ret[i] = NDArrays.getBlasWrapper().iamax(output.getRow(i));
        return ret;
    }


    /**
     * Classify input
     * @param x the input (can either be a matrix or vector)
     * If it's a matrix, each row is considered an example
     * and associated rows are classified accordingly.
     * Each row will be the likelihood of a label given that example
     * @return a probability distribution for each row
     */
    public  INDArray output(INDArray x) {
        if(x == null)
            throw new IllegalArgumentException("No null input allowed");

        this.input = x;
        INDArray preOutput = preOutput(x);
        INDArray ret = activationFunction.apply(preOutput);
        applyDropOutIfNecessary(ret);
        return ret;


    }

    public boolean isConcatBiases() {
        return concatBiases;
    }

    public void setConcatBiases(boolean concatBiases) {
        this.concatBiases = concatBiases;
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

    public  INDArray getInput() {
        return input;
    }

    public  void setInput(INDArray input) {
        this.input = input;
    }

    public  INDArray getLabels() {
        return labels;
    }

    public  void setLabels(INDArray labels) {
        this.labels = labels;
    }

    public  INDArray getW() {
        return W;
    }

    public  void setW(INDArray w) {
        W = w;
    }

    public  INDArray getB() {
        return b;
    }

    public  void setB(INDArray b) {
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



    public  boolean isNormalizeByInputRows() {
        return normalizeByInputRows;
    }



    public  void setNormalizeByInputRows(boolean normalizeByInputRows) {
        this.normalizeByInputRows = normalizeByInputRows;
    }

    public void setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }

    public double getDropOut() {
        return dropOut;
    }

    public void setDropOut(double dropOut) {
        this.dropOut = dropOut;
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

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }


    public boolean isConstrainGradientToUniNorm() {
        return constrainGradientToUniNorm;
    }

    public void setConstrainGradientToUniNorm(boolean constrainGradientToUniNorm) {
        this.constrainGradientToUniNorm = constrainGradientToUniNorm;
    }

    public static class Builder {
        private INDArray W;
        private OutputLayer ret;
        private INDArray b;
        private double l2;
        private int nIn;
        private int nOut;
        private INDArray input;
        private INDArray labels;
        private boolean useRegualarization;
        private ActivationFunction activationFunction = Activations.softmax();
        private boolean useAdaGrad = true;
        private boolean normalizeByInputRows = true;
        private OptimizationAlgorithm optimizationAlgorithm;
        private LossFunction lossFunction = LossFunction.MCXENT;
        private double dropOut = 0;
        private boolean concatBiases = false;
        private boolean constrainGradientToUniNorm = false;
        private WeightInit weightInit;


        public Builder withLabels(INDArray labels)  {
            this.labels = labels;
            return this;
        }

        /**
         * Weight initialization scheme
         * @param weightInit the weight initialization scheme
         * @return
         */
        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        public Builder constrainGradientToUniNorm(boolean constrainGradientToUniNorm) {
            this.constrainGradientToUniNorm = constrainGradientToUniNorm;
            return this;
        }

        public Builder concatBiases(boolean concatBiases) {
            this.concatBiases = concatBiases;
            return this;
        }


        public Builder withDropout(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }

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

        public Builder withWeights(INDArray W) {
            this.W = W;
            return this;
        }

        public Builder withBias(INDArray b) {
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
            ret = new OutputLayer(input, labels,nIn, nOut,weightInit);
            if(W != null)
                ret.W = W;
            if(b != null)
                ret.b = b;
            ret.weightinit = weightInit;
            ret.constrainGradientToUniNorm = constrainGradientToUniNorm;
            ret.dropOut = dropOut;
            ret.optimizationAlgorithm = optimizationAlgorithm;
            ret.normalizeByInputRows = normalizeByInputRows;
            ret.useRegularization = useRegualarization;
            ret.l2 = l2;
            ret.useAdaGrad = useAdaGrad;
            ret.lossFunction = lossFunction;
            ret.activationFunction = activationFunction;
            ret.concatBiases = concatBiases;
            return ret;
        }

    }

}
