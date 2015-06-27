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

import java.io.Serializable;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Classifier;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;
import org.nd4j.linalg.util.LinAlgExceptions;

import static org.nd4j.linalg.ops.transforms.Transforms.log;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;


/**
 * Output layer with different objective
 * incooccurrences for different objectives.
 * This includes classification as well as prediction
 * @author Adam Gibson
 *
 */
public class OutputLayer extends BaseLayer implements Serializable,Classifier {

    private static final long serialVersionUID = -7065564817460914364L;
    //current input and label matrices
    private INDArray labels;

    public OutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public OutputLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    /**
     * Objective function:  the specified objective
     * @return the score for the objective
     */

    @Override
    public  double score() {
        LinAlgExceptions.assertRows(input, labels);
        INDArray output  = output(input);
        if(conf.getLossFunction() != LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
            return  LossFunctions.score(labels,conf.getLossFunction(),output,conf.getL2(),conf.isUseRegularization());

        return  -LossFunctions.score(labels,conf.getLossFunction(),output,conf.getL2(),conf.isUseRegularization());


    }

    @Override
    public void setScore() {
        LinAlgExceptions.assertRows(input,labels);
        INDArray output  = output(input);
        score =  LossFunctions.score(labels,conf.getLossFunction(),output,conf.getL2(),conf.isUseRegularization());

    }


    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(),score());
    }


    /**
     * Gets the gradient from one training iteration
     * @return the gradient (bias and weight matrix)
     */
    @Override
    public Gradient gradient() {
        LinAlgExceptions.assertRows(input, labels);


        //input activation
        INDArray netOut = activate(input);
        //difference of outputs
        INDArray dy = netOut.sub(labels);


        INDArray wGradient = getWeightGradient();
        INDArray bGradient = dy.mean(0);
        Gradient g = new DefaultGradient();

        g.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,wGradient);
        g.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, bGradient);

        return g;

    }



    private INDArray getWeightGradient() {
        INDArray z = output(input);

        switch (conf.getLossFunction()) {
            case MCXENT:
                INDArray preOut = preOutput(input);
                //input activation
                INDArray pYGivenX = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax",preOut),1);
                //difference of outputs
                INDArray dy = pYGivenX.sub(labels);
                return input.transpose().mmul(dy);

            case XENT:
                INDArray xEntDiff = labels.sub(z);
                return input.transpose().mmul(xEntDiff.div(z.mul(z.rsub(1))));
            case MSE:
                INDArray mseDelta = labels.sub(z);
                return input.transpose().mmul(mseDelta.neg());
            case EXPLL:
                return input.transpose().mmul(labels.rsub(1).divi(z));
            case RMSE_XENT:
                INDArray rmseXentDiff = labels.sub(z);
                INDArray squaredrmseXentDiff = pow(rmseXentDiff, 2.0);
                INDArray sqrt = sqrt(squaredrmseXentDiff);
                return input.transpose().mmul(sqrt);
            case SQUARED_LOSS:
                return input.transpose().mmul(pow(labels.sub(z),2));
            case NEGATIVELOGLIKELIHOOD:
                return input.transpose().mmul(log(z).negi());


        }

        throw new IllegalStateException("Invalid loss function");

    }


    @Override
    public INDArray activate(INDArray input) {
        return output(input);
    }

    @Override
    public INDArray activate() {
        return output(input);
    }

    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    @Override
    public double score(DataSet data) {
        return score(data.getFeatureMatrix(), data.getLabels());
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
    public double score(INDArray examples, INDArray labels) {
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
        return labels.columns();
    }

    @Override
    public void fit(DataSetIterator iter) {
        while(iter.hasNext())
            fit(iter.next());
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
        this.input = examples.dup();
        applyDropOutIfNecessary(this.input);
        this.labels = labels;
        Solver solver = new Solver.Builder()
                .configure(conf())
                .listeners(getIterationListeners())
                .model(this).build();
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
     * Set the parameters for this model.
     * This expects a linear ndarray which then be unpacked internally
     * relative to the expected ordering of the model
     *
     * @param params the parameters for the model
     */
    @Override
    public void setParams(INDArray params) {
        INDArray wParams = params.get(NDArrayIndex.interval(0, conf.getNIn() * conf.getNOut()));
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        W.assign(wParams);
        INDArray bias = getParam(DefaultParamInitializer.BIAS_KEY);
        int biasBegin = params.length() - bias.length();
        int biasEnd = params.length();
        INDArray biasAssign = params.get(NDArrayIndex.interval(biasBegin, biasEnd));
        bias.assign(biasAssign);
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


    /**
     * Classify input
     * @param x the input (can either be a matrix or vector)
     * If it's a matrix, each row is considered an example
     * and associated rows are classified accordingly.
     * Each row will be the likelihood of a label given that example
     * @return a probability distribution for each row
     */
    public  INDArray output(INDArray x) {
        return output(x,false);

    }

    /**
     * Classify input
     * @param x the input (can either be a matrix or vector)
     * If it's a matrix, each row is considered an example
     * and associated rows are classified accordingly.
     * Each row will be the likelihood of a label given that example
     * @return a probability distribution for each row
     */
    public  INDArray output(INDArray x,boolean test) {
        if(x == null)
            throw new IllegalArgumentException("No null input allowed");

        INDArray preOutput = preOutput(x);
        if(conf.getActivationFunction().equals("softmax")) {
            INDArray ret = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", preOutput), 1);
            return ret;
        }

        this.input = x.dup();
        if(!test)
            applyDropOutIfNecessary(input());

        return super.activate();

    }

    public  INDArray getLabels() {
        return labels;
    }

    public  void setLabels(INDArray labels) {
        this.labels = labels;
    }


}
