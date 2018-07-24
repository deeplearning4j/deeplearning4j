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

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;


/**
 * Space to channels utility layer for convolutional input types.
 * <p>
 * This operation takes 4D array in, in either NCHW or NHWC format, and moves data from spatial dimensions (HW)
 * to channels (C) for given blockSize
 * <p></p>
 * Example:
 * blockSize = 4
 * dataFormat = "NCHW"
 * input shape =  [128, 16, 16, 3]
 * output shape = [128, 16/4, 16/4, 3*4*4]
 *
 *
 *
 * @author Max Pumperla
 */
@Slf4j
public class SpaceToDepth extends AbstractLayer<org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer> {

    public SpaceToDepth(NeuralNetConfiguration conf) {
        super(conf);
    }

    public SpaceToDepth(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    private int getBlockSize() {
        return layerConf().getBlockSize();
    }

    private int isNHWC() {return layerConf().getDataFormat().equals(SpaceToDepthLayer.DataFormat.NHWC)? 1: 0;}

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int inDepth = (int) input.size(1);
        int inH = (int) input.size(2);
        int inW = (int) input.size(3);

        INDArray outEpsilon = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, new int[]{1, miniBatch * inDepth * inH * inW}, 'c');
        INDArray reshapedEpsilon;

        if (isNHWC() == 1) {
            reshapedEpsilon = outEpsilon.reshape('c', miniBatch, inH, inW, inDepth);
        } else {
            reshapedEpsilon = outEpsilon.reshape('c', miniBatch, inDepth, inH, inW);
        }

        Gradient gradient = new DefaultGradient();

        int blockSize = getBlockSize();

        CustomOp op = DynamicCustomOp.builder("depth_to_space")
                .addInputs(epsilon)
                .addIntegerArguments(blockSize, isNHWC())
                .addOutputs(reshapedEpsilon)
                .build();
        Nd4j.getExecutioner().exec(op);

        reshapedEpsilon = backpropDropOutIfPresent(reshapedEpsilon);
        return new Pair<>(gradient, reshapedEpsilon);
    }

    protected INDArray preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, null);

        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to space to channels with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape [minibatchSize, channels, inputHeight, inputWidth]. "
                    + layerId());
        }

        if (preOutput != null && forBackprop) {
            return preOutput;
        }

        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int depth = (int) input.size(1);
        int inH = (int) input.size(2);
        int inW = (int) input.size(3);

        int blockSize = getBlockSize();

        int outH = inH / blockSize;
        int outW = inW / blockSize;
        int outDepth = depth * blockSize * blockSize;

        INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, new int[]{1, miniBatch * outDepth * outH * outW}, 'c');
        INDArray reshapedOut;
        if (isNHWC() == 1) {
            reshapedOut = out.reshape('c', miniBatch, outH, outW,  outDepth);
        } else {
            reshapedOut = out.reshape('c', miniBatch, outDepth, outH, outW);
        }

        CustomOp op = DynamicCustomOp.builder("space_to_depth")
                .addInputs(input)
                .addIntegerArguments(blockSize, isNHWC())
                .addOutputs(reshapedOut)
                .build();
        Nd4j.getExecutioner().exec(op);

        return reshapedOut;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return preOutput(training, false, workspaceMgr);
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
    public int numParams() {
        return 0;
    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void accumulateScore(double accum) {
        throw new UnsupportedOperationException(layerId());
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