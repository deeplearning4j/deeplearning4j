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
package org.deeplearning4j.nn.layers.recurrent;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.List;

/**
 * Recurrent Neural Network Loss Layer.<br>
 * Handles calculation of gradients etc for various objective functions.<br>
 * NOTE: Unlike {@link RnnOutputLayer} this RnnLossLayer does not have any parameters - i.e., there is no time
 * distributed dense component here. Consequently, the output activations size is equal to the input size.<br>
 * Input and output activations are same as other RNN layers: 3 dimensions with shape
 * [miniBatchSize,nIn,timeSeriesLength] and [miniBatchSize,nOut,timeSeriesLength] respectively.
 *
 * @author Alex Black
 * @see RnnOutputLayer
 */
public class RnnLossLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.RnnLossLayer> implements IOutputLayer {
    @Setter @Getter protected INDArray labels;

    public RnnLossLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (input.rank() != 3)
            throw new UnsupportedOperationException(
                            "Input is not rank 3. Got input with rank " + input.rank() + " " + layerId());
        if (labels == null)
            throw new IllegalStateException("Labels are not set (null)");

        INDArray input2d = TimeSeriesUtils.reshape3dTo2d(input, workspaceMgr, ArrayType.BP_WORKING_MEM);
        INDArray labels2d = TimeSeriesUtils.reshape3dTo2d(labels, workspaceMgr, ArrayType.BP_WORKING_MEM);
        INDArray maskReshaped;
        if(this.maskArray != null){
            if(this.maskArray.rank() == 3){
                maskReshaped = TimeSeriesUtils.reshapePerOutputTimeSeriesMaskTo2d(this.maskArray, workspaceMgr, ArrayType.BP_WORKING_MEM);
            } else {
                maskReshaped = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(this.maskArray, workspaceMgr, ArrayType.BP_WORKING_MEM);
            }
        } else {
            maskReshaped = null;
        }

        // delta calculation
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray delta2d = lossFunction.computeGradient(labels2d, input2d.dup(input2d.ordering()), layerConf().getActivationFn(), maskReshaped);

        INDArray delta3d = TimeSeriesUtils.reshape2dTo3d(delta2d, input.size(0), workspaceMgr, ArrayType.ACTIVATION_GRAD);

        // grab the empty gradient
        Gradient gradient = new DefaultGradient();
        return new Pair<>(gradient, delta3d);
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

    /**{@inheritDoc}
     */
    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        INDArray out = activate(examples, false, null);
        Evaluation eval = new Evaluation();
        eval.evalTimeSeries(labels, out, maskArray);
        return eval.f1();
    }

    @Override
    public int numLabels() {
        return labels.size(1);
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
        return Type.RECURRENT;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if (input.rank() != 3)
            throw new UnsupportedOperationException(
                            "Input must be rank 3. Got input with rank " + input.rank() + " " + layerId());

        return layerConf().getActivationFn().getActivation(workspaceMgr.dup(ArrayType.ACTIVATIONS, input), training);
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
        this.maskArray = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray, LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT);   //TODO
        this.maskState = currentMaskState;

        return null; //Last layer in network
    }

    @Override
    public boolean needsLabels() {
        return true;
    }

    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray input2d = TimeSeriesUtils.reshape3dTo2d(input, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray labels2d = TimeSeriesUtils.reshape3dTo2d(labels, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray maskReshaped;
        if(this.maskArray != null){
            if(this.maskArray.rank() == 3){
                maskReshaped = TimeSeriesUtils.reshapePerOutputTimeSeriesMaskTo2d(this.maskArray, workspaceMgr, ArrayType.FF_WORKING_MEM);
            } else {
                maskReshaped = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(this.maskArray, workspaceMgr, ArrayType.FF_WORKING_MEM);
            }
        } else {
            maskReshaped = null;
        }

        ILossFunction lossFunction = layerConf().getLossFn();

        double score = lossFunction.computeScore(labels2d, input2d.dup(), layerConf().getActivationFn(), maskReshaped,false);
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
        //For RNN: need to sum up the score over each time step before returning.

        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());

        INDArray input2d = TimeSeriesUtils.reshape3dTo2d(input, workspaceMgr, ArrayType.FF_WORKING_MEM);
        INDArray labels2d = TimeSeriesUtils.reshape3dTo2d(labels, workspaceMgr, ArrayType.FF_WORKING_MEM);

        INDArray maskReshaped;
        if(this.maskArray != null){
            if(this.maskArray.rank() == 3){
                maskReshaped = TimeSeriesUtils.reshapePerOutputTimeSeriesMaskTo2d(this.maskArray, workspaceMgr, ArrayType.FF_WORKING_MEM);
            } else {
                maskReshaped = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(this.maskArray, workspaceMgr, ArrayType.FF_WORKING_MEM);
            }
        } else {
            maskReshaped = null;
        }

        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray scoreArray =
                lossFunction.computeScoreArray(labels2d, input2d, layerConf().getActivationFn(), maskReshaped);
        //scoreArray: shape [minibatch*timeSeriesLength, 1]
        //Reshape it to [minibatch, timeSeriesLength] then sum over time step

        INDArray scoreArrayTs = TimeSeriesUtils.reshapeVectorToTimeSeriesMask(scoreArray, input.size(0));
        INDArray summedScores = scoreArrayTs.sum(1);

        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if (l1l2 != 0.0) {
            summedScores.addi(l1l2);
        }

        return summedScores;
    }
}
