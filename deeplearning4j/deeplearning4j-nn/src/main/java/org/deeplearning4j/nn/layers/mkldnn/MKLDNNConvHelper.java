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

import org.deeplearning4j.nn.conf.CNN2DFormat;
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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2DDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Collections;
import java.util.Map;

/**
 * MKL-DNN Convolution (2d) helper
 *
 * @author Alex Black
 */
public class MKLDNNConvHelper implements ConvolutionHelper {

    protected OpContext context;
    protected OpContext contextBwd;

    public MKLDNNConvHelper(DataType dataType){

    }

    @Override
    public boolean checkSupported() {
        return BaseMKLDNNHelper.mklDnnEnabled();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray weights, INDArray bias, INDArray delta, int[] kernel, int[] strides, int[] pad,
                                                     INDArray biasGradView, INDArray weightGradView, IActivation afn, ConvolutionLayer.AlgoMode mode,
                                                     ConvolutionLayer.BwdFilterAlgo bwdFilterAlgo, ConvolutionLayer.BwdDataAlgo bwdDataAlgo, ConvolutionMode convolutionMode,
                                                     int[] dilation, CNN2DFormat format, LayerWorkspaceMgr workspaceMgr) {
        if(input.dataType() != DataType.FLOAT || weights.dataType() != DataType.FLOAT)
            return null;    //MKL-DNN only supports floating point dtype

        //Note: conv2d op expects [kH, kW, iC, oC] weights... DL4J conv uses [oC, iC, kH, kW]
        INDArray weightsPermute = weights.permute(2,3,1,0);
        INDArray weightGradViewPermute = weightGradView.permute(2,3,1,0);

        int hDim = 2;
        int wDim = 3;
        if(format == CNN2DFormat.NHWC){
            hDim = 1;
            wDim = 2;
        }

        if (convolutionMode == ConvolutionMode.Same) {
            pad = ConvolutionUtils.getSameModeTopLeftPadding(new int[]{(int)delta.size(hDim), (int)delta.size(wDim)}, new int[] {(int) input.size(hDim), (int) input.size(wDim)},
                    kernel, strides, dilation);
        }

        if(contextBwd == null){
            contextBwd = Nd4j.getExecutioner().buildContext();
            contextBwd.setIArguments(kernel[0], kernel[1],
                    strides[0], strides[1],
                    pad[0], pad[1],
                    dilation[0], dilation[1],
                    ArrayUtil.fromBoolean(convolutionMode == ConvolutionMode.Same),
                    format == CNN2DFormat.NCHW ? 0 : 1   //0=NCHW, 1=NHWC
            );
        };

        INDArray gradAtInput = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape());

        INDArray[] inputsArr = biasGradView == null ? new INDArray[]{input, weightsPermute, delta} : new INDArray[]{input, weightsPermute, bias, delta};
        INDArray[] outputArr = biasGradView == null ? new INDArray[]{gradAtInput, weightGradViewPermute} : new INDArray[]{gradAtInput, weightGradViewPermute, biasGradView};
        contextBwd.purge();
        for( int i=0; i<inputsArr.length; i++ ){
            contextBwd.setInputArray(i, inputsArr[i]);
        }
        for( int i=0; i<outputArr.length; i++ ){
            contextBwd.setOutputArray(i, outputArr[i]);
        }

        Conv2DDerivative op = new Conv2DDerivative();
        Nd4j.exec(op, contextBwd);

        Gradient g = new DefaultGradient();
        if(biasGradView != null) {
            g.gradientForVariable().put(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        }
        g.gradientForVariable().put(ConvolutionParamInitializer.WEIGHT_KEY, weightGradView);

        return new Pair<>(g, gradAtInput);
    }

    @Override
    public INDArray preOutput(INDArray input, INDArray weights, INDArray bias, int[] kernel, int[] strides, int[] pad,
                              ConvolutionLayer.AlgoMode mode, ConvolutionLayer.FwdAlgo fwdAlgo, ConvolutionMode convolutionMode,
                              int[] dilation, CNN2DFormat format, LayerWorkspaceMgr workspaceMgr) {
        if(input.dataType() != DataType.FLOAT || weights.dataType() != DataType.FLOAT)
            return null;    //MKL-DNN only supports floating point dtype


        int hDim = 2;
        int wDim = 3;
        if(format == CNN2DFormat.NHWC){
            hDim = 1;
            wDim = 2;
        }

        int inH = (int)input.size(hDim);
        int inW = (int)input.size(wDim);
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation, format); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides, dilation);
        } else {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation, format); //Also performs validation
        }

        if(context == null ){
            context = Nd4j.getExecutioner().buildContext();
            context.setIArguments(kernel[0], kernel[1],
                    strides[0], strides[1],
                    pad[0], pad[1],
                    dilation[0], dilation[1],
                    ArrayUtil.fromBoolean(convolutionMode == ConvolutionMode.Same),
                    format == CNN2DFormat.NCHW ? 0 : 1   //0=NCHW, 1=NHWC
            );
        };

        int outDepth = (int) weights.size(0);
        long[] outShape = (format == CNN2DFormat.NCHW) ? new long[]{input.size(0), outDepth, outSize[0], outSize[1]} : new long[]{input.size(0), outSize[0], outSize[1], outDepth};
        INDArray out = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.dataType(), outShape);

        //Note: conv2d op expects [kH, kW, iC, oC] weights... DL4J conv uses [oC, iC, kH, kW]
        weights = weights.permute(2,3,1,0);

        INDArray[] inputsArr = bias == null ? new INDArray[]{input, weights} : new INDArray[]{input, weights, bias};
        context.purge();
        for( int i=0; i<inputsArr.length; i++ ){
            context.setInputArray(i, inputsArr[i]);
        }

        context.setOutputArray(0, out);
        Conv2D op = new Conv2D();
        Nd4j.exec(op, context);

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
