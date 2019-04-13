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
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingHelper;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Collections;
import java.util.Map;

/**
 * MKL-DNN Subsampling (2d) helper
 *
 * @author Alex Black
 */
public class MKLDNNSubsamplingHelper implements SubsamplingHelper {

    protected OpContext context;

    @Override
    public boolean checkSupported() {
        return BaseMKLDNNHelper.mklDnnEnabled();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, int[] kernel, int[] strides, int[] pad, PoolingType poolingType, ConvolutionMode convolutionMode, int[] dilation, LayerWorkspaceMgr workspaceMgr) {
        if(poolingType == PoolingType.SUM || poolingType == PoolingType.PNORM)
            return null;

        INDArray gradAtInput = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape());

        if (convolutionMode == ConvolutionMode.Same) {
            pad = ConvolutionUtils.getSameModeTopLeftPadding(new int[]{(int)epsilon.size(2), (int)epsilon.size(3)}, new int[] {(int)input.size(2), (int)input.size(3)}, kernel, strides, dilation);
        }

        input = input.dup();
        epsilon = epsilon.dup();

        Pooling2DConfig conf = Pooling2DConfig.builder()
                .isSameMode(convolutionMode == ConvolutionMode.Same)
                .kH(kernel[0]).kW(kernel[1])
                .sH(strides[0]).sW(strides[1])
                .dH(dilation[0]).dW(dilation[1])
                .pH(pad[0]).pW(pad[1])
                .isNHWC(false)
                .build();

        switch (poolingType){
            case MAX:
                conf.setType(Pooling2D.Pooling2DType.MAX);
                break;
            case AVG:
                conf.setType(Pooling2D.Pooling2DType.AVG);
                break;
        }

        Pooling2DDerivative d = Pooling2DDerivative.derivativeBuilder()
                .config(conf)
                .arrayInputs(new INDArray[]{input, epsilon})
                .arrayOutputs(new INDArray[]{gradAtInput})
                .build();

        Nd4j.exec(d);
        return new Pair<Gradient,INDArray>(new DefaultGradient(), gradAtInput);
    }

    @Override
    public INDArray activate(INDArray input, boolean training, int[] kernel, int[] strides, int[] pad, PoolingType poolingType, ConvolutionMode convolutionMode, int[] dilation, LayerWorkspaceMgr workspaceMgr) {
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {(int)input.size(2), (int)input.size(3)}, kernel, strides, dilation);
        } else {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
        }

        long[] outShape = new long[]{input.size(0), input.size(1), outSize[0], outSize[1]};
        INDArray output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.dataType(), outShape);

        if(context == null){
            context = Nd4j.getExecutioner().buildContext();
            context.setIArguments(
                    kernel[0], kernel[1],
                    strides[0], strides[1],
                    pad[0], pad[1],
                    dilation[0], dilation[1],
                    ArrayUtil.fromBoolean(convolutionMode == ConvolutionMode.Same),
                    0,  //Extra - not used?
                    0); //0 = NCHW
        }

        DynamicCustomOp op;
        switch (poolingType){
            case MAX:
                op = new MaxPooling2D();
                break;
            case AVG:
                op = new AvgPooling2D();
                break;
            case SUM:
            case PNORM:
            default:
                return null;
        }

        context.getInputArrays().clear();
        context.getOutputArrays().clear();

        context.setInputArray(0, input);
        context.setOutputArray(0, output);

        Nd4j.exec(op, context);
        return output;
    }

    @Override
    public Map<String, Long> helperMemoryUse() {
        return Collections.emptyMap();
    }
}
