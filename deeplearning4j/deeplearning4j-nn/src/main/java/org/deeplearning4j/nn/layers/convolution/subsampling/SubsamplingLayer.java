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

package org.deeplearning4j.nn.layers.convolution.subsampling;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

@Slf4j
public class SubsamplingLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.SubsamplingLayer> {

    protected int helperCountFail = 0;
    protected ConvolutionMode convolutionMode;
    public SubsamplingLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
        initializeHelper();
        this.convolutionMode =
                ((org.deeplearning4j.nn.conf.layers.SubsamplingLayer) conf.getLayer()).getConvolutionMode();
    }

    void initializeHelper() {

    }

    @Override
    public double calcRegularizationScore(boolean backpropOnlyParams) {
        return 0;
    }

    @Override
    public Type type() {
        return Type.SUBSAMPLING;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        INDArray input = this.input.castTo(dataType);
        if(epsilon.dataType() != dataType)
            epsilon = epsilon.castTo(dataType);

        CNN2DFormat dataFormat = layerConf().getCnn2dDataFormat();
        int hIdx = 2;
        int wIdx = 3;
        if(dataFormat == CNN2DFormat.NHWC) {
            hIdx = 1;
            wIdx = 2;
        }

        int inH = (int)input.size(hIdx);
        int inW = (int)input.size(wIdx);

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] dilation = layerConf().getDilation();

        int[] pad;
        int[] outSizeFwd = new int[]{(int)epsilon.size(hIdx), (int)epsilon.size(wIdx)};    //NCHW
        boolean same = convolutionMode == ConvolutionMode.Same;
        if (same) {
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSizeFwd, new int[] {inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
        }


        //subsampling doesn't have weights and thus gradients are not calculated for this layer
        //only scale and reshape epsilon
        Gradient retGradient = new DefaultGradient();


        INDArray epsAtInput = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape(), 'c');
        DynamicCustomOp.DynamicCustomOpsBuilder b;
        int extra = 0;
        switch (layerConf().getPoolingType()) {
            case MAX:
                b = DynamicCustomOp.builder("maxpool2d_bp");
                break;
            case AVG:
                b = DynamicCustomOp.builder("avgpool2d_bp");
                if(layerConf().isAvgPoolIncludePadInDivisor()){
                    //Mostly this is a legacy case - beta4 and earlier models.
                    extra = 1;    //Divide by "number present" excluding padding
                } else {
                    //Default behaviour
                    extra = 0;    //Divide by kH*kW not "number present"
                }

                break;
            case PNORM:
                b = DynamicCustomOp.builder("pnormpool2d_bp");
                extra = layerConf().getPnorm();
                b.addFloatingPointArguments(layerConf().getEps());
                break;
            default:
                throw new UnsupportedOperationException("Pooling mode not supported in SubsamplingLayer: " + layerConf().getPoolingType());
        }

        b.addInputs(input, epsilon)
                .addOutputs(epsAtInput)
                .addIntegerArguments(kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                        (same ? 1 : 0), extra,
                        dataFormat == CNN2DFormat.NCHW ? 0 : 1);  //0 = NCHW, 1=NHWC

        Nd4j.exec(b.build());

        return new Pair<>(retGradient, epsAtInput);
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        //Normally we would apply dropout first. However, dropout on subsampling layers is not something that users typically expect
        // consequently, we'll skip it here

        //Input validation: expect rank 4 matrix
        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape " + layerConf().getCnn2dDataFormat().dimensionNames() + ". "
                    + layerId());
        }

        INDArray input = this.input.castTo(dataType);
        boolean same = convolutionMode == ConvolutionMode.Same;
        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] dilation = layerConf().getDilation();
        int[] pad = layerConf().getPadding();

        DynamicCustomOp.DynamicCustomOpsBuilder b;
        int extra = 0;
        switch (layerConf().getPoolingType()) {
            case MAX:
                b = DynamicCustomOp.builder("maxpool2d");
                break;
            case AVG:
                b = DynamicCustomOp.builder("avgpool2d");
                if(layerConf().isAvgPoolIncludePadInDivisor()) {
                    //Mostly this is a legacy case - beta4 and earlier models.
                    extra = 1;    //Divide by "number present" excluding padding
                } else {
                    //Default behaviour
                    extra = 0;    //Divide by kH*kW not "number present"
                }
                break;
            case PNORM:
                b = DynamicCustomOp.builder("pnormpool2d");
                extra = layerConf().getPnorm();
                break;
            default:
                throw new UnsupportedOperationException("Not supported: " + layerConf().getPoolingType());
        }

        b.addInputs(input)
                .addIntegerArguments(kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                        (same ? 1 : 0), extra,
                        layerConf().getCnn2dDataFormat() == CNN2DFormat.NCHW ? 0 : 1);  //0: NCHW, 1=NHWC

        DynamicCustomOp build = b.build();
        long[] shape = build.calculateOutputShape().get(0).getShape();

        INDArray output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.dataType(), shape, 'c');
        build.addOutputArgument(output);

        Nd4j.exec(build);

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
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {}

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

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        if (maskArray == null) {
            //For same mode (with stride 1): output activations size is always same size as input activations size -> mask array is same size
            return new Pair<>(maskArray, currentMaskState);
        }

        INDArray outMask = ConvolutionUtils.cnn2dMaskReduction(maskArray, layerConf().getKernelSize(), layerConf().getStride(),
                layerConf().getPadding(), layerConf().getDilation(), layerConf().getConvolutionMode());
        return super.feedForwardMaskArray(outMask, currentMaskState, minibatchSize);
    }
}
