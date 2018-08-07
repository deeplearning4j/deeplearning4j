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

package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

/**Recurrent Neural Network Output Layer.<br>
 * Handles calculation of gradients etc for various objective functions.<br>
 * Functionally the same as OutputLayer, but handles output and label reshaping
 * automatically.<br>
 * Input and output activations are same as other RNN layers: 3 dimensions with shape
 * [miniBatchSize,nIn,timeSeriesLength] and [miniBatchSize,nOut,timeSeriesLength] respectively.
 * @author Alex Black
 * @see BaseOutputLayer, OutputLayer
 */
public class RnnOutputLayer extends BaseOutputLayer<org.deeplearning4j.nn.conf.layers.RnnOutputLayer> {

    public RnnOutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public RnnOutputLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (input.rank() != 3)
            throw new UnsupportedOperationException(
                            "Input is not rank 3. Got input with rank " + input.rank() + " " + layerId());
        INDArray inputTemp = input;
        this.input = TimeSeriesUtils.reshape3dTo2d(input, workspaceMgr, ArrayType.BP_WORKING_MEM);
        Pair<Gradient, INDArray> gradAndEpsilonNext = super.backpropGradient(epsilon, workspaceMgr);
        this.input = inputTemp;
        INDArray epsilon2d = gradAndEpsilonNext.getSecond();

        // FIXME: int cast
        INDArray epsilon3d = TimeSeriesUtils.reshape2dTo3d(epsilon2d, (int) input.size(0), workspaceMgr, ArrayType.ACTIVATION_GRAD);

        weightNoiseParams.clear();

        epsilon3d = backpropDropOutIfPresent(epsilon3d);
        return new Pair<>(gradAndEpsilonNext.getFirst(), epsilon3d);
    }

    /**{@inheritDoc}
     */
    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        if (examples.rank() == 3)
            examples = TimeSeriesUtils.reshape3dTo2d(examples, LayerWorkspaceMgr.noWorkspaces(), ArrayType.ACTIVATIONS);
        if (labels.rank() == 3)
            labels = TimeSeriesUtils.reshape3dTo2d(labels, LayerWorkspaceMgr.noWorkspaces(), ArrayType.ACTIVATIONS);
        return super.f1Score(examples, labels);
    }

    public INDArray getInput() {
        return input;
    }

    @Override
    public Layer.Type type() {
        return Layer.Type.RECURRENT;
    }

    @Override
    protected INDArray preOutput2d(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if (input.rank() == 3) {
            //Case when called from RnnOutputLayer
            INDArray inputTemp = input;
            input = TimeSeriesUtils.reshape3dTo2d(input, LayerWorkspaceMgr.noWorkspaces(), ArrayType.FF_WORKING_MEM);
            INDArray out = super.preOutput(training, workspaceMgr);
            this.input = inputTemp;
            return out;
        } else {
            //Case when called from BaseOutputLayer
            INDArray out = super.preOutput(training, workspaceMgr);
            return out;
        }
    }

    @Override
    protected INDArray getLabels2d(LayerWorkspaceMgr workspaceMgr, ArrayType arrayType) {
        if (labels.rank() == 3)
            return TimeSeriesUtils.reshape3dTo2d(labels, workspaceMgr, arrayType);
        return labels;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (input.rank() != 3)
            throw new UnsupportedOperationException(
                            "Input must be rank 3. Got input with rank " + input.rank() + " " + layerId());
        INDArray b = getParamWithNoise(DefaultParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, training, workspaceMgr);

        applyDropOutIfNecessary(training, workspaceMgr);
        INDArray input2d = TimeSeriesUtils.reshape3dTo2d(input, LayerWorkspaceMgr.noWorkspaces(), ArrayType.FF_WORKING_MEM);

        //INDArray act2d = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(),
        //        input2d.mmul(W).addiRowVector(b)));
        INDArray act2d = layerConf().getActivationFn().getActivation(input2d.mmul(W).addiRowVector(b), training);
        if (maskArray != null) {
            if(!maskArray.isColumnVectorOrScalar() || Arrays.equals(maskArray.shape(), act2d.shape())){
                //Per output masking
                act2d.muli(maskArray);
            } else {
                //Per time step masking
                act2d.muliColumnVector(maskArray);
            }
        }

        // FIXME: int cast
        return TimeSeriesUtils.reshape2dTo3d(act2d, (int) input.size(0), workspaceMgr, ArrayType.ACTIVATIONS);
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        if (maskArray != null) {
            //Two possible cases:
            //(a) per time step masking - rank 2 mask array -> reshape to rank 1 (column vector)
            //(b) per output masking - rank 3 mask array  -> reshape to rank 2 (
            if (maskArray.rank() == 2) {
                this.maskArray = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray, LayerWorkspaceMgr.noWorkspacesImmutable(), ArrayType.INPUT);
            } else if (maskArray.rank() == 3) {
                this.maskArray = TimeSeriesUtils.reshape3dTo2d(maskArray, LayerWorkspaceMgr.noWorkspacesImmutable(), ArrayType.INPUT);
            } else {
                throw new UnsupportedOperationException(
                                "Invalid mask array: must be rank 2 or 3 (got: rank " + maskArray.rank() + ", shape = "
                                                + Arrays.toString(maskArray.shape()) + ") " + layerId());
            }
        } else {
            this.maskArray = null;
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {

        //If the *input* mask array is present and active, we should use it to mask the output
        if (maskArray != null && currentMaskState == MaskState.Active) {
            this.inputMaskArray = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray, LayerWorkspaceMgr.noWorkspacesImmutable(), ArrayType.INPUT);
            this.inputMaskArrayState = currentMaskState;
        } else {
            this.inputMaskArray = null;
            this.inputMaskArrayState = null;
        }

        return null; //Last layer in network
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
        INDArray preOut = preOutput2d(false, workspaceMgr);

        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray scoreArray =
                        lossFunction.computeScoreArray(getLabels2d(workspaceMgr, ArrayType.FF_WORKING_MEM), preOut,
                                layerConf().getActivationFn(), maskArray);
        //scoreArray: shape [minibatch*timeSeriesLength, 1]
        //Reshape it to [minibatch, timeSeriesLength] then sum over time step

        // FIXME: int cast
        INDArray scoreArrayTs = TimeSeriesUtils.reshapeVectorToTimeSeriesMask(scoreArray, (int) input.size(0));
        INDArray summedScores = scoreArrayTs.sum(1);

        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if (l1l2 != 0.0) {
            summedScores.addi(l1l2);
        }

        return summedScores;
    }
}
