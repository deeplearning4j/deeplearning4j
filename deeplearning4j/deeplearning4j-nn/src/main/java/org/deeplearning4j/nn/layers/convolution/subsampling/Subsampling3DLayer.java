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

package org.deeplearning4j.nn.layers.convolution.subsampling;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.Convolution3DUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;


/**
 * Subsampling 3D layer, used for downsampling a 3D convolution
 *
 * @author Max Pumperla
 */
@Slf4j
public class Subsampling3DLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.Subsampling3DLayer> {

    protected ConvolutionMode convolutionMode;

    public Subsampling3DLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
        this.convolutionMode =
                ((org.deeplearning4j.nn.conf.layers.Subsampling3DLayer) conf.getLayer()).getConvolutionMode();
    }


    @Override
    public double calcRegularizationScore(boolean backpropParamsOnly){
        return 0;
    }

    @Override
    public Type type() {
        return Type.SUBSAMPLING;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        boolean isNCDHW = layerConf().getDataFormat() == Convolution3D.DataFormat.NCDHW;

        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int inChannels = (int) (isNCDHW ? input.size(1) : input.size(4));
        int inD = (int) (isNCDHW ? input.size(2) : input.size(1));
        int inH = (int) (isNCDHW ? input.size(3) : input.size(2));
        int inW = (int) (isNCDHW ? input.size(4) : input.size(3));

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] dilation = layerConf().getDilation();

        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = Convolution3DUtils.get3DOutputSize(
                    input, kernel, strides, null, convolutionMode, dilation, isNCDHW);
            pad = Convolution3DUtils.get3DSameModeTopLeftPadding(
                    outSize, new int[]{inD, inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
        }

        INDArray outEpsilon = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, epsilon.dataType(),
                isNCDHW ? new long[]{miniBatch, inChannels, inD, inH, inW} : new long[]{miniBatch, inD, inH, inW, inChannels}, 'c');


        int[] intArgs = new int[]{
                kernel[0], kernel[1], kernel[2],
                strides[0], strides[1], strides[2],
                pad[0], pad[1], pad[2],
                dilation[0], dilation[1], dilation[2],
                convolutionMode == ConvolutionMode.Same ? 1 : 0,
                0,  //Extra param - 0 = exclude padding for average divisor
                isNCDHW ? 0 : 1
        };

        String opName = layerConf().getPoolingType() == PoolingType.MAX ? "maxpool3dnew_bp" : "avgpool3dnew_bp";

        CustomOp op = DynamicCustomOp.builder(opName)
                .addInputs(input, epsilon)
                .addIntegerArguments(intArgs)
                .addOutputs(outEpsilon)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        Gradient retGradient = new DefaultGradient();
        outEpsilon = backpropDropOutIfPresent(outEpsilon);
        return new Pair<>(retGradient, outEpsilon);
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if (training && !dropoutApplied && layerConf().getIDropout() != null) {
            applyDropOutIfNecessary(true, workspaceMgr);
        }

        boolean isNCDHW = layerConf().getDataFormat() == Convolution3D.DataFormat.NCDHW;

        if (input.rank() != 5) {
            if(isNCDHW){
                throw new DL4JInvalidInputException("Got rank " + input.rank()
                        + " array as input to Subsampling3DLayer with shape " + Arrays.toString(input.shape())
                        + ". Expected rank 5 array with shape [minibatchSize, channels, "
                        + "inputDepth, inputHeight, inputWidth] when dataFormat=NCDHW. "
                        + layerId());
            } else {
                throw new DL4JInvalidInputException("Got rank " + input.rank()
                        + " array as input to Subsampling3DLayer with shape " + Arrays.toString(input.shape())
                        + ". Expected rank 5 array with shape [minibatchSize, inputDepth, inputHeight, inputWidth, channels] when dataFormat=NDHWC. "
                        + layerId());
            }
        }

        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int inChannels = (int) (isNCDHW ? input.size(1) : input.size(4));
        int inD = (int) (isNCDHW ? input.size(2) : input.size(1));
        int inH = (int) (isNCDHW ? input.size(3) : input.size(2));
        int inW = (int) (isNCDHW ? input.size(4) : input.size(3));

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] dilation = layerConf().getDilation();
        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            int[] inShape = new int[]{inD, inH, inW};
            outSize = Convolution3DUtils.get3DOutputSize(
                    input, kernel, strides, null, convolutionMode, dilation, isNCDHW);
            pad = Convolution3DUtils.get3DSameModeTopLeftPadding(outSize, inShape, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
            outSize = Convolution3DUtils.get3DOutputSize(
                    input, kernel, strides, pad, convolutionMode, dilation, isNCDHW);
        }
        int outD = outSize[0];
        int outH = outSize[1];
        int outW = outSize[2];

        String opName = layerConf().getPoolingType() == PoolingType.MAX ? "maxpool3dnew" : "avgpool3dnew";

        INDArray output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.dataType(),
                isNCDHW ? new long[]{miniBatch, inChannels, outD, outH, outW} : new long[]{miniBatch, outD, outH, outW, inChannels}, 'c');

        int[] intArgs = new int[]{
                kernel[0], kernel[1], kernel[2],
                strides[0], strides[1], strides[2],
                pad[0], pad[1], pad[2],
                dilation[0], dilation[1], dilation[2],
                convolutionMode == ConvolutionMode.Same ? 1 : 0,
                0,  //Extra param - 0 = exclude padding for average divisor (only applicable for average pooling)
                isNCDHW ? 0 : 1
        };

        CustomOp op = DynamicCustomOp.builder(opName)
                .addInputs(input)
                .addIntegerArguments(intArgs)
                .addOutputs(output)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        return output;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //no op
    }

    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException("Not supported - no parameters");
    }

    @Override
    public void fit() {

    }

    @Override
    public long numParams() {
        return 0;
    }

    @Override
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
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
