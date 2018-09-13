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

package org.deeplearning4j.nn.layers.convolution;

import lombok.Getter;
import lombok.Setter;
import lombok.val;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.List;

/**
 * 3D Convolutional Neural Network Loss Layer.<br>
 * Handles calculation of gradients etc for various objective functions.<br>
 * NOTE: Cnn3DLossLayer does not have any parameters. Consequently, the output activations size is equal to the input size.<br>
 * Input and output activations are same as other 3D CNN layers: 5 dimensions with shape [miniBatchSize,channels,depth,height,width]<br>
 * Cnn3DLossLayer has support for a built-in activation function (tanh, softmax etc) - if this is not required, set
 * activation function to Activation.IDENTITY. For activations such as softmax, note that this is applied channels-wise:
 * that is, softmax is applied along dimension 1 (channels) for each minibatch, and x/y/z location separately.<br>
 * <br>
 * Note that 3 types of masking are supported: (n=minibatchSize, c=channels, d=depth, h=height, w=width)<br>
 * - Per example masking: Where an example is present or not (and all outputs are masked by it). Mask shape [n,1]<br>
 * - Per x/y/z location masking: where each spatial X/Y/Z location is present or not (all channels at a given x/y/z are masked by it).
 * Mask shape: [n,d,h,w].<br>
 * - Per output masking: Where each output activation value is present or not - mask shape [n,c,d,h,w] (same as output)<br>
 *
 * @author Alex Black
 */
public class Cnn3DLossLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.Cnn3DLossLayer> implements IOutputLayer {
    @Setter
    @Getter
    protected INDArray labels;

    public Cnn3DLossLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (input.rank() != 5)
            throw new UnsupportedOperationException(
                    "Input is not rank 5. Got input with rank " + input.rank() + " " + layerId() + " with shape "
                            + Arrays.toString(input.shape()) + " - expected shape [minibatch,channels,depth,height,width]");
        if (labels == null)
            throw new IllegalStateException("Labels are not set (null)");

        INDArray input2d = ConvolutionUtils.reshape5dTo2d(input, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray labels2d = ConvolutionUtils.reshape5dTo2d(labels, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray maskReshaped = ConvolutionUtils.reshapeMaskIfRequired(maskArray, input, workspaceMgr, ArrayType.FF_WORKING_MEM);

        // delta calculation
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray delta2d = lossFunction.computeGradient(labels2d, input2d.dup(input2d.ordering()), layerConf().getActivationFn(), maskReshaped);
        delta2d = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, delta2d);

        // FIXME: int cast
        INDArray delta5d = ConvolutionUtils.reshape2dTo5d(delta2d, ArrayUtil.toInts(input.shape()), workspaceMgr, ArrayType.ACTIVATION_GRAD);

        // grab the empty gradient
        Gradient gradient = new DefaultGradient();
        return new Pair<>(gradient, delta5d);
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
    public double f1Score(DataSet data) {
        return 0;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        INDArray out = activate(examples, false, null); //TODO
        Evaluation eval = new Evaluation();
        eval.evalTimeSeries(labels, out, maskArray);
        return eval.f1();
    }

    @Override
    public int numLabels() {
        // FIXME: int cast
        return (int) labels.size(1);
    }

    @Override
    public void fit(DataSetIterator iter) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int[] predict(INDArray examples) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<String> predict(DataSet dataSet) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray labelProbabilities(INDArray examples) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(DataSet data) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(INDArray examples, int[] labels) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL3D;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if (input.rank() != 5)
            throw new UnsupportedOperationException(
                    "Input must be rank 5. Got input with rank " + input.rank() + " " + layerId());

        INDArray in = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, input.ordering());
        INDArray input2d = ConvolutionUtils.reshape5dTo2d(in, workspaceMgr, ArrayType.ACTIVATIONS);
        INDArray out2d = layerConf().getActivationFn().getActivation(input2d, training);

        // FIXME: int cast
        return ConvolutionUtils.reshape2dTo5d(out2d, ArrayUtil.toInts(input.shape()), workspaceMgr, ArrayType.ACTIVATIONS);
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        this.maskArray = maskArray;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                                                          int minibatchSize) {
        this.maskArray = maskArray;
        return null; //Last layer in network
    }

    @Override
    public boolean needsLabels() {
        return true;
    }

    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray input2d = ConvolutionUtils.reshape5dTo2d(input, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray labels2d = ConvolutionUtils.reshape5dTo2d(labels, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray maskReshaped = ConvolutionUtils.reshapeMaskIfRequired(maskArray, input, workspaceMgr, ArrayType.FF_WORKING_MEM);

        ILossFunction lossFunction = layerConf().getLossFn();

        double score = lossFunction.computeScore(labels2d, input2d.dup(), layerConf().getActivationFn(), maskReshaped, false);
        score += fullNetworkL1 + fullNetworkL2;
        score /= getInputMiniBatchSize();

        this.score = score;

        return score;
    }

    /**
     * Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetworkL1 L1 regularization term for the entire network (or, 0.0 to not include regularization)
     * @param fullNetworkL2 L2 regularization term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2, LayerWorkspaceMgr workspaceMgr) {
        //For 3D CNN: need to sum up the score over each x/y/z location before returning

        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());

        INDArray input2d = ConvolutionUtils.reshape5dTo2d(input, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray labels2d = ConvolutionUtils.reshape5dTo2d(labels, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray maskReshaped = ConvolutionUtils.reshapeMaskIfRequired(maskArray, input, workspaceMgr, ArrayType.FF_WORKING_MEM);

        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray scoreArray =
                lossFunction.computeScoreArray(labels2d, input2d, layerConf().getActivationFn(), maskReshaped);
        //scoreArray: shape [minibatch*d*h*w, 1]
        //Reshape it to [minibatch, 1, d, h, w] then sum over x/y/z to give [minibatch, 1]

        val newShape = input.shape().clone();
        newShape[1] = 1;

        // FIXME
        INDArray scoreArrayTs = ConvolutionUtils.reshape2dTo5d(scoreArray, ArrayUtil.toInts(newShape), workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray summedScores = scoreArrayTs.sum(1,2,3,4);

        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if (l1l2 != 0.0) {
            summedScores.addi(l1l2);
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, summedScores);
    }
}
