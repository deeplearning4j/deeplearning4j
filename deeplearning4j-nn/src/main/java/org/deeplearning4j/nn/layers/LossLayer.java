/*-
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


import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;


/**
 * LossLayer is a flexible output "layer" that performs a loss function on
 * an input without MLP logic.
 *
 * @author Justin Long (crockpotveggies)
 */
public class LossLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.LossLayer>
                implements Serializable, IOutputLayer {

    //current input and label matrices
    protected INDArray labels;
    @Setter @Getter
    protected INDArray labelMask;

    private transient Solver solver;

    private double fullNetworkL1;
    private double fullNetworkL2;

    private double score;

    public LossLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public void init() {
        //No op
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        //No op
    }

    @Override
    public void setListeners(IterationListener... listeners) {
        //No op
    }

    @Override
    public void addListeners(IterationListener... listener) {
        //No op
    }

    @Override
    public Collection<IterationListener> getListeners() {
        return Collections.emptyList();
    }

    @Override
    public double score(){
        return score;
    }

    /** Compute score after labels and input have been set.
     * @param fullNetworkL1 L1 regularization term for the entire network
     * @param fullNetworkL2 L2 regularization term for the entire network
     * @param training whether score should be calculated at train or test time (this affects things like application of
     *                 dropout, etc)
     * @return score (loss function)
     */
    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training) {
        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        this.fullNetworkL1 = fullNetworkL1;
        this.fullNetworkL2 = fullNetworkL2;
        INDArray input = this.input.get(0);
        INDArray preOut = input;

        ILossFunction lossFunction = layerConf().getLossFn();

        //double score = lossFunction.computeScore(getLabels2d(), preOut, layerConf().getActivationFunction(), this.input.getMask(0), false);
        double score = lossFunction.computeScore(getLabels2d(), preOut, layerConf().getActivationFn(), this.input.getMask(0),
                        false);
        score += fullNetworkL1 + fullNetworkL2;
        score /= getInputMiniBatchSize();

        this.score = score;

        return score;
    }

    /**Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetworkL1 L1 regularization term for the entire network (or, 0.0 to not include regularization)
     * @param fullNetworkL2 L2 regularization term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2) {
        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        INDArray input = this.input.get(0);
        INDArray preOut = input;

        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray scoreArray =
                        lossFunction.computeScoreArray(getLabels2d(), preOut, layerConf().getActivationFn(), this.input.getMask(0));
        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if (l1l2 != 0.0) {
            scoreArray.addi(l1l2);
        }
        return scoreArray;
    }

    @Override
    public void computeGradientAndScore() {
        if (input == null || labels == null)
            return;
        INDArray input = this.input.get(0);

        INDArray preOut = input;
        Gradients pair = getGradientsAndDelta(preOut);
        this.gradient = pair.getParameterGradients();

        score = computeScore(fullNetworkL1, fullNetworkL2, true);
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return null;
    }

    @Override
    public Gradients backpropGradient(Gradients epsilon) {
        return getGradientsAndDelta(input.get(0));
    }


    /** Returns tuple: {Gradient,Delta,Output} given preOut */
    private Gradients getGradientsAndDelta(INDArray preOut) {
        // delta calculation
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray delta = lossFunction.computeGradient(getLabels2d(), preOut, layerConf().getActivationFn(), this.input.getMask(0));

        // grab the empty gradient
        Gradient gradient = new DefaultGradient();

        Gradients g = GradientsFactory.getInstance().create(delta, gradient);
        return backpropPreprocessor(g);
    }

    /**
     * Gets the gradient from one training iteration
     * @return the gradient (bias and weight matrix)
     */
    @Override
    public Gradient gradient() {
        return gradient;
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public void fit(Activations data) {
        throw new UnsupportedOperationException("LossLayer cannot be fit from input activations only (no labels)");
    }

    @Override
    public Activations activate(boolean training) {
        INDArray input = this.input.get(0);
        INDArray z = input;
        INDArray ret = layerConf().getActivationFn().getActivation(z.dup(), training);

        if (this.input.getMask(0) != null) {
            ret.muliColumnVector(this.input.getMask(0));
        }

        return ActivationsFactory.getInstance().create(ret);
    }

    @Override
    public Activations activate(Activations input, boolean training) {
        setInput(input);
        return ActivationsFactory.getInstance().create(output(training));
    }

    public INDArray output(INDArray input, boolean training) {
        setInput(ActivationsFactory.getInstance().create(input));
        return output(training);
    }

    public INDArray output(INDArray input) {
        setInput(ActivationsFactory.getInstance().create(input));
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
    public INDArray output(boolean training) {
        if (input == null) {
            throw new IllegalArgumentException("Cannot perform forward pass with null input " + layerId());
        }
        return activate(training).get(0);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public void fit(DataSetIterator iter) {
        // no-op
    }


    /**
     * Fit the model
     *
     * @param input the examples to classify (one example in each row)
     * @param labels   the example labels(a binary outcome matrix)
     */
    @Override
    public void fit(INDArray input, INDArray labels) {
        throw new UnsupportedOperationException("LossLayer has no parameters and cannot be fit");
    }

    /**
     * Fit the model
     *
     * @param data the data to train on
     */
    @Override
    public void fit(DataSet data) {
        fit(data.getFeatures(), data.getLabels());
    }

    @Override
    public void clear() {
        super.clear();
        if (labels != null) {
            labels.data().destroy();
            labels = null;
        }
        solver = null;
    }

    @Override
    public INDArray getLabels() {
        return labels;
    }

    @Override
    public void setLabels(INDArray labels, INDArray labelMask) {
        this.labels = labels;
        this.labelMask = labelMask;
    }

    protected INDArray getLabels2d() {
        if (labels.rank() > 2) {
            return labels.reshape(labels.size(2), labels.size(1));
        }
        return labels;
    }
}
