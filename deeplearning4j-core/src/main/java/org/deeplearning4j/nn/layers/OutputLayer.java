package org.deeplearning4j.nn.layers;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

import java.io.Serializable;

import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.api.NeuralNetwork.OptimizationAlgorithm;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.OutputLayerGradient;
import org.nd4j.linalg.learning.AdaGrad;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
import org.deeplearning4j.optimize.optimizers.OutputLayerOptimizer;
import org.deeplearning4j.optimize.solvers.StochasticHessianFree;
import org.deeplearning4j.optimize.solvers.VectorizedDeepLearningGradientAscent;
import org.deeplearning4j.optimize.solvers.VectorizedNonZeroStoppingConjugateGradient;


/**
 * Output layer with different objective functions for different objectives.
 * This includes classification as well as prediction
 * @author Adam Gibson
 *
 */
public class OutputLayer extends BaseLayer implements Serializable,Classifier {

    private static final long serialVersionUID = -7065564817460914364L;
    //current input and label matrices
    private INDArray labels;
    private AdaGrad adaGrad,biasAdaGrad;






    public OutputLayer(NeuralNetConfiguration conf,INDArray input, INDArray labels) {
        super(conf,null,null,input);
        this.labels = labels;

        adaGrad = new AdaGrad(conf.getnIn(),conf.getnOut());
        b = Nd4j.zeros(1,conf.getnOut());
        biasAdaGrad = new AdaGrad(b.rows(),b.columns());
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
     * @param learningRate the learning rate to iterate with
     * @param numEpochs the number of epochs
     * @param eval the training evaluator to use for early stopping (where applicable)
     */
    public  void trainTillConvergence(INDArray labels,double learningRate, int numEpochs,TrainingEvaluator eval) {

        this.labels = labels;
        OutputLayerOptimizer opt = new OutputLayerOptimizer(this, learningRate);
        adaGrad.setMasterStepSize(learningRate);
        biasAdaGrad.setMasterStepSize(learningRate);

        if(conf.getOptimizationAlgo() == OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(opt);
            g.setTolerance(1e-3f);
            g.setTrainingEvaluator(eval);
            g.setMaxIterations(numEpochs);
            g.optimize(numEpochs);

        }
        else if(conf.getOptimizationAlgo()  == OptimizationAlgorithm.HESSIAN_FREE) {
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
     * @param learningRate the learning rate to iterate with
     * @param numEpochs the number of epochs
     * @param eval the training evaluator to use for early stopping (where applicable)
     */
    public  void trainTillConvergence(double learningRate, int numEpochs,TrainingEvaluator eval) {
        OutputLayerOptimizer opt = new OutputLayerOptimizer(this, learningRate);
        adaGrad.setMasterStepSize(learningRate);
        biasAdaGrad.setMasterStepSize(learningRate);

        if(conf.getOptimizationAlgo()  == OptimizationAlgorithm.CONJUGATE_GRADIENT) {
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
     * @param learningRate the learning rate to iterate with
     * @param numEpochs the number of epochs
     */
    public  void trainTillConvergence(double learningRate, int numEpochs) {
        trainTillConvergence(learningRate,numEpochs,null);
    }




    /**
     * Objective function:  the specified objective
     * @return the score for the objective
     */

    @Override
    public  float score() {
        LinAlgExceptions.assertRows(input,labels);
        INDArray output  = output(input);
        assert !Nd4j.hasInvalidNumber(output) : "Invalid number on output!";
        if(conf.getLossFunction() != LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
            return  LossFunctions.score(input,conf.getLossFunction(),transform(input),conf.getL2(),conf.isUseRegularization());

        return  LossFunctions.score(labels,conf.getLossFunction(),output,conf.getL2(),conf.isUseRegularization());


    }




    /**
     * Train on the given inputs and labels.
     * This will assign the passed in values
     * as fields to this logistic function for
     * caching.
     * @param x the inputs to iterate on
     * @param y the labels to iterate on
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
    public  org.deeplearning4j.nn.api.Layer clone()  {
        OutputLayer reg = new OutputLayer(conf,W,b);
        if(this.labels != null)
            reg.labels = this.labels.dup();
        reg.biasAdaGrad = this.biasAdaGrad;
        reg.adaGrad = this.adaGrad;
        if(this.input != null)
            reg.input = this.input.dup();
        return    reg;
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
        dy.divi(input.rows());

        INDArray wGradient = getWeightGradient();
        if(conf.isUseAdaGrad())
            wGradient.muli(adaGrad.getLearningRates(wGradient));
        else
            wGradient.muli(lr);

        if(conf.isUseAdaGrad())
            dy.muliRowVector(biasAdaGrad.getLearningRates(dy.mean(0)));
        else
            dy.muli(lr);

        dy.divi(input.rows());


        INDArray bGradient = dy.mean(0);
        if(conf.isConstrainGradientToUnitNorm()) {
            wGradient.divi(wGradient.norm2(Integer.MAX_VALUE));
            bGradient.divi(bGradient.norm2(Integer.MAX_VALUE));
        }


        return new OutputLayerGradient(wGradient,bGradient);


    }


    private INDArray getWeightGradient() {
        INDArray z = output(input);

        switch (conf.getLossFunction()) {
            case MCXENT:
                INDArray preOut = preOutput(input);
                //input activation
                INDArray p_y_given_x = Activations.sigmoid().apply(preOut);
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
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    @Override
    public float score(DataSet data) {
        return score(data.getFeatureMatrix(),data.getLabels());
    }

    /**
     * Returns the f1 score for the given examples.
     * Think of this to be like a percentage right.
     * The higher the number the more it got right.
     * This is on a scale from 0 to 1.
     *
     * @param examples te the examples to classify (one example in each row)
     * @param labels   the true labels
     * @return the scores for each ndarray
     */
    @Override
    public float score(INDArray examples, INDArray labels) {
        Evaluation eval = new Evaluation();
        eval.eval(labels,labelProbabilities(examples));
        return (float) eval.f1();

    }

    /**
     * Returns the number of possible labels
     *
     * @return the number of possible labels for this classifier
     */
    @Override
    public int numLabels() {
        return labels.columns();
    }

    /**
     * Returns the predictions for each example in the dataset
     * @param d the matrix to predict
     * @return the prediction for the dataset
     */
    @Override
    public int[] predict(INDArray d) {
        INDArray output = output(d);
        int[] ret = new int[d.rows()];
        for(int i = 0; i < ret.length; i++)
            ret[i] = Nd4j.getBlasWrapper().iamax(output.getRow(i));
        return ret;
    }

    /**
     * Returns the probabilities for each label
     * for each example row wise
     *
     * @param examples the examples to classify (one example in each row)
     * @return the likelihoods of each example and each label
     */
    @Override
    public INDArray labelProbabilities(INDArray examples) {
        return output(examples);
    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the example labels(a binary outcome matrix)
     */
    @Override
    public void fit(INDArray examples, INDArray labels) {
        trainTillConvergence(examples,labels,conf.getLr(),conf.getNumIterations());
    }

    /**
     * Fit the model
     *
     * @param data the data to train on
     */
    @Override
    public void fit(DataSet data) {
        fit(data.getFeatureMatrix(),data.getLabels());
    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the example labels(a binary outcome matrix)
     * @param params   extra parameters
     */
    @Override
    public void fit(INDArray examples, INDArray labels, Object[] params) {
        fit(examples,labels);
    }

    /**
     * Fit the model
     *
     * @param data   the data to train on
     * @param params extra parameters
     */
    @Override
    public void fit(DataSet data, Object[] params) {
        fit(data);
    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     */
    @Override
    public void fit(INDArray examples, int[] labels) {
        INDArray outcomeMatrix = FeatureUtil.toOutcomeMatrix(labels, numLabels());
        fit(examples,outcomeMatrix);

    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     *                 the number of rows in the example
     * @param params   extra parameters
     */
    @Override
    public void fit(INDArray examples, int[] labels, Object[] params) {
        INDArray labelMatrix = FeatureUtil.toOutcomeMatrix(labels,labels.length);
        fit(examples,labelMatrix);
    }

    /**
     * Iterate once on the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     *                 the number of rows in the example
     * @param params   extra parameters
     */
    @Override
    public void iterate(INDArray examples, int[] labels, Object[] params) {

    }

    /**
     * Transform the data based on the model's output.
     * This can be anything from a number to reconstructions.
     *
     * @param data the data to transform
     * @return the transformed data
     */
    @Override
    public INDArray transform(INDArray data) {
        return preOutput(data);
    }

    /**
     * Returns the coefficients for this classifier as a raveled vector
     *
     * @return a copy of this classifier's params
     */
    @Override
    public INDArray params() {
        return Nd4j.hstack(W.linearView(),b.linearView());
    }

    /**
     * The number of parameters for the model
     *
     * @return the number of parameters for the model
     */
    @Override
    public int numParams() {
        return conf.getnIn() * conf.getnOut() + conf.getnOut();
    }

    /**
     * Set the parameters for this model.
     * This expects a linear ndarray which then be unpacked internally
     * relative to the expected ordering of the model
     *
     * @param params the parameters for the model
     */
    @Override
    public void setParams(INDArray params) {
        INDArray wParams = params.get(NDArrayIndex.interval(0, conf.getnIn() * conf.getnOut()));
        INDArray wLinear = getW().linearView();
        for(int i = 0; i < wParams.length(); i++) {
            wLinear.putScalar(i,wParams.get(i));
        }
        setB(params.get(NDArrayIndex.interval(conf.getnIn() * conf.getnOut(), params.length())));
    }

    /**
     * Fit the model to the given data
     *
     * @param data   the data to fit the model to
     * @param params the params (mixed values)
     */
    @Override
    public void fit(INDArray data, Object[] params) {
        throw new UnsupportedOperationException();

    }

    /**
     * Fit the model to the given data
     *
     * @param data the data to fit the model to
     */
    @Override
    public void fit(INDArray data) {
       throw new UnsupportedOperationException();
    }

    /**
     * Run one iteration
     *
     * @param input  the input to iterate on
     * @param params the extra params for the neural network(k, corruption level, max epochs,...)
     */
    @Override
    public void iterate(INDArray input, Object[] params) {

    }

    @Override
    public String toString() {
        return "OutputLayer{" +
                "labels=" + labels +
                ", adaGrad=" + adaGrad +
                ", biasAdaGrad=" + biasAdaGrad +
                "} " + super.toString();
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
        INDArray ret = conf.getActivationFunction().apply(preOutput);
        applyDropOutIfNecessary(ret);
        return ret;


    }


    public  INDArray getLabels() {
        return labels;
    }

    public  void setLabels(INDArray labels) {
        this.labels = labels;
    }



    public AdaGrad getBiasAdaGrad() {
        return biasAdaGrad;
    }


    public AdaGrad getAdaGrad() {
        return adaGrad;
    }


    public void setAdaGrad(AdaGrad adaGrad) {
        this.adaGrad = adaGrad;
    }

    public void setBiasAdaGrad(AdaGrad biasAdaGrad) {
        this.biasAdaGrad = biasAdaGrad;
    }



    public static class Builder {
        private INDArray W;
        private OutputLayer ret;
        private NeuralNetConfiguration conf;
        private INDArray b;
        private INDArray input;
        private INDArray labels;


        public Builder configure(NeuralNetConfiguration conf) {
            this.conf = conf;
            return this;
        }


        public Builder input(INDArray input) {
            this.input = input;
            return this;
        }


        public Builder withLabels(INDArray labels)  {
            this.labels = labels;
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


        public OutputLayer build() {
            ret = new OutputLayer(conf,input, labels);
            if(W != null)
                ret.W = W;
            if(b != null)
                ret.b = b;
            ret.conf = conf;
            return ret;
        }

    }

}
