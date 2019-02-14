/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.nn.layers.mkldnn;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.convolution.ConvolutionHelper;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.Map;

public class MKLDNNConvHelper implements ConvolutionHelper {
    @Override
    public boolean checkSupported() {
        return BaseMKLDNNHelper.mklDnnEnabled();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray weights, INDArray bias, INDArray delta, int[] kernel, int[] strides, int[] pad,
                                                     INDArray biasGradView, INDArray weightGradView, IActivation afn, ConvolutionLayer.AlgoMode mode,
                                                     ConvolutionLayer.BwdFilterAlgo bwdFilterAlgo, ConvolutionLayer.BwdDataAlgo bwdDataAlgo, ConvolutionMode convolutionMode,
                                                     int[] dilation, LayerWorkspaceMgr workspaceMgr) {
        //Note: conv2d op expects [kH, kW, iC, oC] weights... DL4J conv uses [oC, iC, kH, kW]
        INDArray weightsPermute = weights.permute(2,3,1,0);
        INDArray weightGradViewPermute = weightGradView.permute(2,3,1,0);

        if (convolutionMode == ConvolutionMode.Same) {
            pad = ConvolutionUtils.getSameModeTopLeftPadding(new int[]{(int)delta.size(2), (int)delta.size(3)}, new int[] {(int) input.size(2), (int) input.size(3)},
                    kernel, strides, dilation);
        }

        Conv2DConfig conf = Conv2DConfig.builder()
                .dataFormat(Conv2DConfig.NCHW)
                .kH(kernel[0]).kW(kernel[1])
                .sH(strides[0]).sW(strides[1])
                .pH(pad[0]).pW(pad[1])
                .dH(dilation[0]).dH(dilation[1])
                .isSameMode(convolutionMode == ConvolutionMode.Same)
                .build();

        INDArray gradAtInput = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape());

        INDArray[] inputsArr = biasGradView == null ? new INDArray[]{input, weightsPermute, delta} : new INDArray[]{input, weightsPermute, bias, delta};
        INDArray[] outputArr = biasGradView == null ? new INDArray[]{gradAtInput, weightGradViewPermute} : new INDArray[]{gradAtInput, weightGradViewPermute, biasGradView};
        DynamicCustomOp op = Conv2DDerivative.derivativeBuilder()
                .config(conf)
                .inputArrays(inputsArr)
                .outputs(outputArr)
                .build();
        Nd4j.exec(op);

        Gradient g = new DefaultGradient();
        if(biasGradView != null) {
            g.gradientForVariable().put(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        }
        g.gradientForVariable().put(ConvolutionParamInitializer.WEIGHT_KEY, weightGradView);

        return new Pair<>(g, gradAtInput);
    }

    @Override
    public INDArray preOutput(INDArray input, INDArray weights, INDArray bias, int[] kernel, int[] strides, int[] pad, ConvolutionLayer.AlgoMode mode, ConvolutionLayer.FwdAlgo fwdAlgo, ConvolutionMode convolutionMode, int[] dilation, LayerWorkspaceMgr workspaceMgr) {

        int inH = (int)input.size(2);
        int inW = (int)input.size(3);
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides, dilation);
        } else {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
        }

        int outDepth = (int) weights.size(0);
        INDArray out = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.dataType(), input.size(0), outDepth, outSize[0], outSize[1]);

        //Note: conv2d op expects [kH, kW, iC, oC] weights... DL4J conv uses [oC, iC, kH, kW]
        weights = weights.permute(2,3,1,0);

        Conv2DConfig conf = Conv2DConfig.builder()
                .dataFormat(Conv2DConfig.NCHW)
                .kH(kernel[0]).kW(kernel[1])
                .sH(strides[0]).sW(strides[1])
                .pH(pad[0]).pW(pad[1])
                .dH(dilation[0]).dH(dilation[1])
                .isSameMode(convolutionMode == ConvolutionMode.Same)
                .build();

        INDArray[] inputsArr = bias == null ? new INDArray[]{input, weights} : new INDArray[]{input, weights, bias};
        DynamicCustomOp op = Conv2D.builder()
                .config(conf)
                .inputArrays(inputsArr)
                .outputs(new INDArray[]{out})
                .build();
        Nd4j.exec(op);

        return out;
    }

    @Override
    public INDArray activate(INDArray z, IActivation afn, boolean training) {
        return afn.getActivation(z, training);
    }

    @Override
    public Map<String, Long> helperMemoryUse() {
        return Collections.emptyMap();
    }
}
