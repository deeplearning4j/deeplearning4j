/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.layers.convolution;

import lombok.Getter;
import lombok.Setter;
import lombok.val;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;
import java.util.List;

public class CnnLossLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.CnnLossLayer> implements IOutputLayer {
    @Setter
    @Getter
    protected INDArray labels;

    public CnnLossLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (input.rank() != 4)
            throw new UnsupportedOperationException(
                    "Input is not rank 4. Got input with rank " + input.rank() + " " + layerId() + " with shape "
                            + Arrays.toString(input.shape()) + " - expected shape " + layerConf().getFormat().dimensionNames());
        if (labels == null)
            throw new IllegalStateException("Labels are not set (null)");

        Preconditions.checkState(input.equalShapes(labels), "Input and label arrays do not have same shape: %ndShape vs. %ndShape", input, labels);

        CNN2DFormat format = layerConf().getFormat();
        INDArray input2d = ConvolutionUtils.reshape4dTo2d(input, format, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray labels2d = ConvolutionUtils.reshape4dTo2d(labels, format, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray maskReshaped = ConvolutionUtils.reshapeMaskIfRequired(maskArray, input, format, workspaceMgr, ArrayType.FF_WORKING_MEM);

        // delta calculation
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray delta2d = lossFunction.computeGradient(labels2d, input2d.dup(input2d.ordering()), layerConf().getActivationFn(), maskReshaped);
        delta2d = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, delta2d);

        INDArray delta4d = ConvolutionUtils.reshape2dTo4d(delta2d, input.shape(), format, workspaceMgr, ArrayType.ACTIVATION_GRAD);

        // grab the empty gradient
        Gradient gradient = new DefaultGradient();
        return new Pair<>(gradient, delta4d);
    }

    @Override
    public double calcRegularizationScore(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double f1Score(DataSet data) {
        return f1Score(data.getFeatures(), data.getLabels());
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
        return Type.CONVOLUTIONAL;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if (input.rank() != 4)
            throw new UnsupportedOperationException(
                    "Input must be rank 4 with shape " + layerConf().getFormat().dimensionNames() +
                            ". Got input with rank " + input.rank() + " " + layerId());

        CNN2DFormat format = layerConf().getFormat();

        INDArray in = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, input.ordering());
        INDArray input2d = ConvolutionUtils.reshape4dTo2d(in, format, workspaceMgr, ArrayType.ACTIVATIONS);
        INDArray out2d = layerConf().getActivationFn().getActivation(input2d, training);
        return ConvolutionUtils.reshape2dTo4d(out2d, input.shape(), format, workspaceMgr, ArrayType.ACTIVATIONS);
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
    public double computeScore(double fullNetRegTerm, boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray input2d = ConvolutionUtils.reshape4dTo2d(input, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray labels2d = ConvolutionUtils.reshape4dTo2d(labels, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray maskReshaped = ConvolutionUtils.reshapeMaskIfRequired(maskArray, input, layerConf().getFormat(), workspaceMgr, ArrayType.FF_WORKING_MEM);

        ILossFunction lossFunction = layerConf().getLossFn();

        double score = lossFunction.computeScore(labels2d, input2d.dup(), layerConf().getActivationFn(), maskReshaped, false);
        score /= getInputMiniBatchSize();
        score += fullNetRegTerm;
        this.score = score;
        return score;
    }

    /**
     * Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetRegTerm Regularization score term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    @Override
    public INDArray computeScoreForExamples(double fullNetRegTerm, LayerWorkspaceMgr workspaceMgr) {
        //For CNN: need to sum up the score over each x/y location before returning

        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());

        CNN2DFormat format = layerConf().getFormat();

        INDArray input2d = ConvolutionUtils.reshape4dTo2d(input, format, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray labels2d = ConvolutionUtils.reshape4dTo2d(labels, format, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray maskReshaped = ConvolutionUtils.reshapeMaskIfRequired(maskArray, input, format, workspaceMgr, ArrayType.FF_WORKING_MEM);

        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray scoreArray =
                lossFunction.computeScoreArray(labels2d, input2d, layerConf().getActivationFn(), maskReshaped);
        //scoreArray: shape [minibatch*h*w, 1]
        //Reshape it to [minibatch, 1, h, w] then sum over x/y to give [minibatch, 1]

        val newShape = input.shape().clone();
        newShape[1] = 1;

        INDArray scoreArrayTs = ConvolutionUtils.reshape2dTo4d(scoreArray, newShape, format, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray summedScores = scoreArrayTs.sum(1, 2, 3).reshape(scoreArrayTs.size(0), 1);

        if (fullNetRegTerm != 0.0) {
            summedScores.addi(fullNetRegTerm);
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, summedScores);
    }
}
