/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.layers;


import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.FeatureUtil;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.io.Serializable;
import java.util.ArrayList;
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

    private transient Solver solver;

    private double fullNetworkL1;
    private double fullNetworkL2;

    public LossLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public LossLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    /** Compute score after labels and input have been set.
     * @param fullNetworkL1 L1 regularization term for the entire network
     * @param fullNetworkL2 L2 regularization term for the entire network
     * @param training whether score should be calculated at train or test time (this affects things like application of
     *                 dropout, etc)
     * @return score (loss function)
     */
    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        this.fullNetworkL1 = fullNetworkL1;
        this.fullNetworkL2 = fullNetworkL2;
        INDArray preOut = input;

        ILossFunction lossFunction = layerConf().getLossFn();

        //double score = lossFunction.computeScore(getLabels2d(), preOut, layerConf().getActivationFunction(), maskArray, false);
        double score = lossFunction.computeScore(getLabels2d(), preOut, layerConf().getActivationFn(), maskArray,
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
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2, LayerWorkspaceMgr workspaceMgr) {
        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        INDArray preOut = input;

        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray scoreArray =
                        lossFunction.computeScoreArray(getLabels2d(), preOut, layerConf().getActivationFn(), maskArray);
        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if (l1l2 != 0.0) {
            scoreArray.addi(l1l2);
        }
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, scoreArray);
    }

    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
        if (input == null || labels == null)
            return;

        INDArray preOut = input;
        Pair<Gradient, INDArray> pair = getGradientsAndDelta(preOut, workspaceMgr);
        this.gradient = pair.getFirst();

        score = computeScore(fullNetworkL1, fullNetworkL2, true, workspaceMgr);
    }

    @Override
    protected void setScoreWithZ(INDArray z) {
        throw new RuntimeException("Not supported " + layerId());
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        return getGradientsAndDelta(input, workspaceMgr);
    }


    /** Returns tuple: {Gradient,Delta,Output} given preOut */
    private Pair<Gradient, INDArray> getGradientsAndDelta(INDArray preOut, LayerWorkspaceMgr workspaceMgr) {
        // delta calculation
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray delta = lossFunction.computeGradient(getLabels2d(), preOut, layerConf().getActivationFn(), maskArray);

        // grab the empty gradient
        Gradient gradient = new DefaultGradient();

        delta = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, delta);
        return new Pair<>(gradient, delta);
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
    public Type type() {
        return Type.FEED_FORWARD;
    }

    @Override
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray z = input;
        INDArray ret = layerConf().getActivationFn().getActivation(z.dup(), training);

        if (maskArray != null) {
            ret.muliColumnVector(maskArray);
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, ret);
    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
        setInput(input, workspaceMgr);
        return activate(training, workspaceMgr);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public INDArray params() {
        return null;
    }



    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    @Override
    public double f1Score(DataSet data) {
        return f1Score(data.getFeatures(), data.getLabels());
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
    public double f1Score(INDArray examples, INDArray labels) {
        Evaluation eval = new Evaluation();
        eval.eval(labels, labelProbabilities(examples));
        return eval.f1();
    }

    /**
     * Returns the number of possible labels
     *
     * @return the number of possible labels for this classifier
     */
    @Override
    public int numLabels() {
        // FIXME: int cast
        return (int) labels.size(1);
    }

    @Override
    public void fit(DataSetIterator iter) {
        // no-op
    }

    /**
     * Returns the predictions for each example in the dataset
     * @param input the matrix to predict
     * @return the prediction for the dataset
     */
    @Override
    public int[] predict(INDArray input) {
        INDArray output = activate(input, false, LayerWorkspaceMgr.noWorkspacesImmutable());
        int[] ret = new int[input.rows()];
        for (int i = 0; i < ret.length; i++)
            ret[i] = Nd4j.getBlasWrapper().iamax(output.getRow(i));
        return ret;
    }

    /**
     * Return predicted label names
     *
     * @param dataSet to predict
     * @return the predicted labels for the dataSet
     */
    @Override
    public List<String> predict(DataSet dataSet) {
        int[] intRet = predict(dataSet.getFeatures());
        List<String> ret = new ArrayList<>();
        for (int i : intRet) {
            ret.add(i, dataSet.getLabelName(i));
        }
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
        return activate(examples, false, LayerWorkspaceMgr.noWorkspacesImmutable());
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

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     */
    @Override
    public void fit(INDArray examples, int[] labels) {
        INDArray outcomeMatrix = FeatureUtil.toOutcomeMatrix(labels, numLabels());
        fit(examples, outcomeMatrix);

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
    public boolean needsLabels() {
        return true;
    }

    public void setLabels(INDArray labels) {
        this.labels = labels;
    }

    protected INDArray getLabels2d() {
        if (labels.rank() > 2) {
            return labels.reshape(labels.size(2), labels.size(1));
        }
        return labels;
    }

}
