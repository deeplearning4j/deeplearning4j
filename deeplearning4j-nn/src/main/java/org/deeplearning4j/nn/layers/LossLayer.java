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
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;


/**
 * LossLayer is a flexible output "layer" that performs a loss function on
 * an input without MLP logic.
 *
 * @author Justin Long (crockpotveggies)
 */
public class LossLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.LossLayer> implements IOutputLayer {
    private static final IActivation IDENTITY = new ActivationIdentity();

    //current input and label matrices
    protected INDArray labels;
    @Setter @Getter
    protected INDArray labelMask;

    private double score;

    public LossLayer(org.deeplearning4j.nn.conf.layers.LossLayer conf) {
        super(conf);
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
    public double computeScore(Activations layerOutput, Activations labels, double fullNetworkL1, double fullNetworkL2, boolean training) {

        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without network output and labels: " + layerId());

        ILossFunction lossFunction = layerConf().getLossFn();

        //For scoring: (preout + activation fn) == (output + identity)
        double score = lossFunction.computeScore(labels.get(0), layerOutput.get(0), IDENTITY, labelMask, false);
        score += fullNetworkL1 + fullNetworkL2;
        score /= labels.get(0).size(0);

        this.score = score;

        clear();
        return score;
    }

    /**Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetworkL1 L1 regularization term for the entire network (or, 0.0 to not include regularization)
     * @param fullNetworkL2 L2 regularization term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    @Override
    public INDArray computeScoreForExamples(Activations layerOutput, Activations labels, double fullNetworkL1, double fullNetworkL2) {
        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());

        ILossFunction lossFunction = layerConf().getLossFn();
        //For scoring: (preout + activation fn) == (output + identity)
        INDArray scoreArray = lossFunction.computeScoreArray(labels.get(0), layerOutput.get(0), IDENTITY, labelMask);
        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if (l1l2 != 0.0) {
            scoreArray.addi(l1l2);
        }

        return scoreArray;
    }

    @Override
    public Gradients backpropGradient(Gradients epsilon) {
        return getGradientsAndDelta(input.get(0));
    }


    /** Returns tuple: {Gradient,Delta,Output} given preOut */
    private Gradients getGradientsAndDelta(INDArray preOut) {
        // delta calculation
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray delta = lossFunction.computeGradient(getLabels(), preOut, layerConf().getActivationFn(), this.input.getMask(0));

        // grab the empty gradient
        Gradient gradient = new DefaultGradient();

        Gradients g = GradientsFactory.getInstance().create(delta, gradient);
        return backpropPreprocessor(g);
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
    public void clear() {
        super.clear();
        if (labels != null) {
            labels.data().destroy();
            labels = null;
        }
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

    @Override
    public void setLabels(Activations labels){
        if(labels == null){
            setLabels(null, null);
        } else {
            if(labels.size() != 1){
                throw new IllegalArgumentException("Cannot set labels: must be of size (# arrays) 1. Got labels size: " + labels.size());
            }
            setLabels(labels.get(0), labels.getMask(0));
        }
    }
}
