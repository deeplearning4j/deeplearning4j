/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;


/**
 * Space to batch utility layer for convolutional input types.
 * <p>
 * Does a 2-dimensional space to batch operation, i.e. ransforms data from a tensor from 2 spatial dimensions into batch dimension
 * according to the "blocks" specified (a vector of length 2). Afterwards the spatial dimensions are optionally padded,
 * as specified in "padding", a tensor of dim (2, 2), denoting the padding range.
 * <p>
 * Example:
 * input:         [[[[1], [2]], [[3], [4]]]]
 * input shape:   [1, 2, 2, 1]
 * blocks:        [2, 2]
 * padding:       [[0, 0], [0, 0]]
 * <p>
 * output:        [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
 * output shape:  [4, 1, 1, 1]
 *
 * @author Max Pumperla
 */
@Slf4j
public class SpaceToBatch extends AbstractLayer<org.deeplearning4j.nn.conf.layers.SpaceToBatchLayer> {

    public SpaceToBatch(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    private int[] getBlocks() {
        return layerConf().getBlocks();
    }

    private int[][] getPadding() {
        return layerConf().getPadding();
    }

    private INDArray getBlocksArray() {
        int[] intBlocks = layerConf().getBlocks();
        return Nd4j.createFromArray(intBlocks);
    }

    private INDArray getPaddingArray() {
        int[][] intPad = layerConf().getPadding();
        return Nd4j.createFromArray(intPad);
    }


    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        INDArray input = this.input.castTo(dataType);   //Cast to network dtype if required (no-op if already correct type)

        boolean nchw = layerConf().getFormat() == CNN2DFormat.NCHW;

        INDArray outEpsilon = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape(), 'c');

        Gradient gradient = new DefaultGradient();

        INDArray epsilonNHWC = nchw ? epsilon.permute(0, 2, 3, 1) : epsilon;
        INDArray outEpsilonNHWC = nchw ? outEpsilon.permute(0, 2, 3, 1) : outEpsilon;

        CustomOp op = DynamicCustomOp.builder("batch_to_space_nd")
                .addInputs(epsilonNHWC, getBlocksArray(), getPaddingArray())
                .addOutputs(outEpsilonNHWC)
                .callInplace(false)
                .build();
        Nd4j.exec(op);

        outEpsilon = backpropDropOutIfPresent(outEpsilon);
        return new Pair<>(gradient, outEpsilon);
    }

    protected INDArray preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, null);

        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to space to batch with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape " + layerConf().getFormat().dimensionNames() + ". "
                    + layerId());
        }

        if (preOutput != null && forBackprop) {
            return preOutput;
        }

        boolean nchw = layerConf().getFormat() == CNN2DFormat.NCHW;

        long inMiniBatch = input.size(0);
        long depth = input.size(nchw ? 1 : 3);
        long inH = input.size(nchw ? 2 : 1);
        long inW = input.size(nchw ? 3 : 2);

        int[] blocks = getBlocks();
        int[][] padding = getPadding();

        long paddedH = inH + padding[0][0] + padding[0][1];
        long paddedW = inW + padding[1][0] + padding[1][1];

        long outH = paddedH / blocks[0];
        long outW = paddedW / blocks[1];
        long outMiniBatch = inMiniBatch * blocks[0] * blocks[1];

        long[] outShape = nchw ? new long[]{outMiniBatch, depth, outH, outW} : new long[]{outMiniBatch, outH, outW, depth};

        INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, input.dataType(), outShape, 'c');

        INDArray inNHWC = nchw ? input.permute(0, 2, 3, 1) : input;
        INDArray outNHWC = nchw ? out.permute(0, 2, 3, 1) : out;

        CustomOp op = DynamicCustomOp.builder("space_to_batch_nd")
                .addInputs(inNHWC, getBlocksArray(), getPaddingArray())
                .addOutputs(outNHWC)
                .build();
        Nd4j.exec(op);

        return out;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return preOutput(training, false, workspaceMgr);
    }


    @Override
    public double calcRegularizationScore(boolean backpropParamsOnly){
        return 0;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //No op
    }

    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException("Not supported - no parameters");
    }

    @Override
    public long numParams() {
        return 0;
    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void update(INDArray gradient, String paramType) {

    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        return params();
    }

    @Override
    public void setParams(INDArray params) {
    }

}
