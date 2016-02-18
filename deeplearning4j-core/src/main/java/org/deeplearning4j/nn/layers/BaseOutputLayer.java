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

package org.deeplearning4j.nn.layers;

import java.io.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Classifier;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossCalculation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;
import org.nd4j.linalg.util.LinAlgExceptions;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;


/**
 * Output layer with different objective
 * incooccurrences for different objectives.
 * This includes classification as well as prediction
 * @author Adam Gibson
 *
 */
public abstract class BaseOutputLayer<LayerConfT extends org.deeplearning4j.nn.conf.layers.BaseOutputLayer>
        extends BaseLayer<LayerConfT> implements Serializable,Classifier {

    //current input and label matrices
    protected INDArray labels;

    private transient Solver solver;

    private double fullNetworkL1;
    private double fullNetworkL2;

    public BaseOutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public BaseOutputLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    /** Compute score after labels and input have been set.
     * @param fullNetworkL1 L1 regularization term for the entire network
     * @param fullNetworkL2 L2 regularization term for the entire network
     * @param training whether score should be calculated at train or test time (this affects things like application of
     *                 dropout, etc)
     * @return score (loss function)
     */
    public double computeScore( double fullNetworkL1, double fullNetworkL2, boolean training) {
        if( input == null || labels == null )
            throw new IllegalStateException("Cannot calculate score without input and labels");
        this.fullNetworkL1 = fullNetworkL1;
        this.fullNetworkL2 = fullNetworkL2;
        INDArray preOut = preOutput2d(training);
        //special case: softmax
        if (layerConf().getActivationFunction().equals("softmax")) {
            setScore(null,preOut);
        } else {
            INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), preOut));
            setScoreWithZ(output);
        }
        return score;
    }

    /**Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetworkL1 L1 regularization term for the entire network (or, 0.0 to not include regularization)
     * @param fullNetworkL2 L2 regularization term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2){
        if( input == null || labels == null )
            throw new IllegalStateException("Cannot calculate score without input and labels");
        INDArray preOut = preOutput2d(false);
        INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), preOut.dup()));

        return LossCalculation.builder()
                .l1(fullNetworkL1).l2(fullNetworkL2)
                .labels(getLabels2d()).z(output)
                .preOut(preOut).activationFn(conf().getLayer().getActivationFunction())
                .lossFunction(layerConf().getLossFunction())
                .useRegularization(conf.isUseRegularization())
                .mask(maskArray).build().scoreExamples();
    }

    @Override
    public void computeGradientAndScore() {
        if(input == null || labels == null)
            return;

        INDArray preOut = preOutput2d(true);
        Triple<Gradient,INDArray,INDArray> triple = getGradientsAndDelta(preOut);
        this.gradient = triple.getFirst();
        setScore(triple.getThird(), preOut);
    }

    @Override
    protected void setScoreWithZ(INDArray z) {
        setScore(z, null);
    }

    private void setScore(INDArray z, INDArray preOut ){
        if (layerConf().getLossFunction() == LossFunctions.LossFunction.CUSTOM) {
            LossFunction create = Nd4j.getOpFactory().createLossFunction(layerConf().getCustomLossFunction(), input, z);
            create.exec();
            score = create.getFinalResult().doubleValue();
        }
        else {
            score = LossCalculation.builder()
                    .l1(fullNetworkL1).l2(fullNetworkL2)
                    .labels(getLabels2d()).z(z)
                    .preOut(preOut).activationFn(conf().getLayer().getActivationFunction())
                    .lossFunction(layerConf().getLossFunction())
                    .miniBatch(conf.isMiniBatch()).miniBatchSize(getInputMiniBatchSize())
                    .useRegularization(conf.isUseRegularization())
                    .mask(maskArray).build().score();
        }
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(),score());
    }

    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {
        Triple<Gradient,INDArray,INDArray> triple = getGradientsAndDelta(preOutput2d(true));	//Returns Gradient and delta^(this), not Gradient and epsilon^(this-1)
        INDArray delta = triple.getSecond();

        INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(delta.transpose()).transpose();
        return new Pair<>(triple.getFirst(),epsilonNext);
    }

    /**
     * Gets the gradient from one training iteration
     * @return the gradient (bias and weight matrix)
     */
    @Override
    public Gradient gradient() {
        LinAlgExceptions.assertRows(input, getLabels2d());
        return gradient;

    }

    /** Returns tuple: {Gradient,Delta,Output} given preOut */
    private Triple<Gradient,INDArray,INDArray> getGradientsAndDelta(INDArray preOut) {
        INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), preOut.dup()));
        INDArray outSubLabels = output.sub(getLabels2d());
        Gradient gradient = new DefaultGradient();

        if(maskArray != null){
            //Masking on gradients. Mask values are 0 or 1. If 0: no output -> no error for this example
            outSubLabels.muliColumnVector(maskArray);
        }

        Triple<Gradient,INDArray,INDArray> triple;
        switch (layerConf().getLossFunction()) {
            case MCXENT:	//cross-entropy (multi-class, with one-hot encoding)
                gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, input.transpose().mmul(outSubLabels));
                gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, outSubLabels.sum(0));
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            case XENT: // cross-entropy (single binary output variable)
                gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, input.transpose().mmul(outSubLabels.div(output.mul(output.rsub(1)))));
                gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, outSubLabels.sum(0));
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            case MSE: // mean squared error
                INDArray delta = outSubLabels.mul(derivativeActivation(preOut));
                gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, input.transpose().mmul(delta));
                gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, delta.sum(0));
                triple = new Triple<>(gradient,delta,output);
                break;

            case EXPLL: // exponential logarithmic
                gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, input.transpose().mmul(labels.rsub(1).divi(output)));
                gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, outSubLabels.sum(0));
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            case RMSE_XENT: // root mean squared error cross entropy
                INDArray squaredrmseXentDiff = pow(outSubLabels, 2.0);
                INDArray sqrt = sqrt(squaredrmseXentDiff);
                gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, input.transpose().mmul(sqrt));
                gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, outSubLabels.sum(0));
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            case SQUARED_LOSS:
                gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, input.transpose().mmul(pow(outSubLabels,2)));
                gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, outSubLabels.sum(0));
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            case NEGATIVELOGLIKELIHOOD: // multi-class cross-entropy
                gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, input.transpose().mmul(outSubLabels));
                gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, outSubLabels.sum(0));
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            default:
                throw new IllegalStateException("Invalid loss function: " + layerConf().getLossFunction());
        }


        return triple;
    }


    @Override
    public INDArray activate(INDArray input, boolean training) {
        setInput(input);
        return output(training);
    }

    @Override
    public INDArray activate(INDArray input) {
        setInput(input);
        return output(true);
    }

    @Override
    public INDArray activate() {
        return output(false);
    }

    public  INDArray output(INDArray input, boolean training) {
        setInput(input);
        return output(training);
    }

    public  INDArray output(INDArray input) {
        setInput(input);
        return output(false);
    }

    /**
     * Classify input
     * @param training determines if its training
     * the input (can either be a matrix or vector)
     * If it's a matrix, each row is considered an example
     * and associated rows are classified accordingly.
     * Each row will be the likelihood of a label given that example
     * @return a probability distribution for each row
     */
    public  INDArray output(boolean training) {
        if(input == null)
            throw new IllegalArgumentException("No null input allowed");

        INDArray preOutput = preOutput2d(training);
        if(conf.getLayer().getActivationFunction().equals("softmax")) {
            SoftMax softMax = new SoftMax(preOutput);
            softMax.exec(1);
            INDArray z = softMax.z();
            if(maskArray != null){
                z.muliColumnVector(maskArray);
            }
            return z;
        }

        if(training)
            applyDropOutIfNecessary(training);

        return super.activate(true);
    }


    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    public double f1Score(DataSet data) {
        return f1Score(data.getFeatureMatrix(), data.getLabels());
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
    public double f1Score(INDArray examples, INDArray labels) {
        Evaluation eval = new Evaluation();
        eval.eval(labels,labelProbabilities(examples));
        return  eval.f1();

    }

    /**
     * Returns the number of possible labels
     *
     * @return the number of possible labels for this classifier
     */
    @Override
    public int numLabels() {
        return labels.size(1);
    }

    @Override
    public void fit(DataSetIterator iter) {
        while(iter.hasNext())
            fit(iter.next());
    }

    /**
     * Returns the predictions for each example in the dataset
     * @param input the matrix to predict
     * @return the prediction for the dataset
     */
    @Override
    public int[] predict(INDArray input) {
        INDArray output = output(input);
        int[] ret = new int[input.rows()];
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
     * @param input the examples to classify (one example in each row)
     * @param labels   the example labels(a binary outcome matrix)
     */
    @Override
    public void fit(INDArray input, INDArray labels) {
        setInput(input);
        setLabels(labels);
        applyDropOutIfNecessary(true);
        if( solver == null ){
            solver = new Solver.Builder()
                    .configure(conf())
                    .listeners(getListeners())
                    .model(this).build();
        }
        solver.optimize();
    }

    /**
     * Fit the model
     *
     * @param data the data to train on
     */
    @Override
    public void fit(DataSet data) {
        fit(data.getFeatureMatrix(), data.getLabels());
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

    @Override
    public void clear() {
        super.clear();
        if(labels != null) {
            labels.data().destroy();
            labels = null;
        }
        solver = null;
    }

    /**
     * Fit the model to the given data
     *
     * @param data the data to fit the model to
     */
    @Override
    public void fit(INDArray data) {
        //no-op

    }

    @Override
    public void iterate(INDArray input) {
        throw new UnsupportedOperationException();
    }


    public  INDArray getLabels() {
        return labels;
    }

    public  void setLabels(INDArray labels) {
        this.labels = labels;
    }

    protected INDArray preOutput2d(boolean training){
        return preOutput(training);
    }

    protected INDArray output2d(INDArray input){
        return output(input);
    }

    protected INDArray getLabels2d(){
        if(labels.rank() > 2) {
            return labels.reshape(labels.size(2),labels.size(1));
        }
        return labels;
    }

}
