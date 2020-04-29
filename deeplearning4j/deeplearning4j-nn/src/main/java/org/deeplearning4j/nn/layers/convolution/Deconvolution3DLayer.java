/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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

import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.Deconvolution3D;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DeconvolutionParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;

import java.util.Arrays;

/**
 * 3D deconvolution layer implementation.
 *
 * Deconvolutions are also known as transpose convolutions or fractionally strided convolutions.
 * In essence, deconvolutions swap forward and backward pass with regular 3D convolutions.
 *
 * See the paper by Matt Zeiler for details:
 * <a href="http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf">http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf</a>
 *
 * For an intuitive guide to convolution arithmetic and shapes, see:
 * <a href="https://arxiv.org/abs/1603.07285v1">https://arxiv.org/abs/1603.07285v1</a>
 *
 *
 * @author Alex Black
 */
public class Deconvolution3DLayer extends BaseLayer<Deconvolution3D> {

    public Deconvolution3DLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (input.rank() != 5) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to Deconvolution3DLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 5 array with shape [minibatchSize, channels, inputHeight, inputWidth, inputDepth] or" +
                    " [minibatchSize, inputHeight, inputWidth, inputDepth, channels]. " + layerId());
        }

        INDArray weights = getParamWithNoise(DeconvolutionParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        Convolution3D.DataFormat df = layerConf().getDataFormat();
        ConvolutionMode cm = layerConf().getConvolutionMode();

        int[] dilation = layerConf().getDilation();
        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad = layerConf().getPadding();

        INDArray biasGradView = gradientViews.get(DeconvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(DeconvolutionParamInitializer.WEIGHT_KEY);

        INDArray outEps = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, weights.dataType(), input.shape(), 'c');

        Integer sameMode = (layerConf().getConvolutionMode() == ConvolutionMode.Same) ? 1 : 0;

        int[] args = new int[] {
                kernel[0], kernel[1], kernel[2], strides[0], strides[1], strides[2],
                pad[0], pad[1], pad[2], dilation[0], dilation[1], dilation[2], sameMode,
                df == Convolution3D.DataFormat.NCDHW ? 0 : 1
        };

        INDArray delta;
        IActivation afn = layerConf().getActivationFn();
        INDArray preOutput = preOutput(true, workspaceMgr);
        delta = afn.backprop(preOutput, epsilon).getFirst();

        INDArray[] opInputs;
        INDArray[] opOutputs;
        if(layerConf().hasBias()){
            INDArray bias = getParamWithNoise(DeconvolutionParamInitializer.BIAS_KEY, true, workspaceMgr);
            opInputs = new INDArray[]{input, weights, bias, delta};
            opOutputs = new INDArray[]{outEps, weightGradView, biasGradView};
        } else {
            opInputs = new INDArray[]{input, weights, delta};
            opOutputs = new INDArray[]{outEps, weightGradView};
        }
        CustomOp op = DynamicCustomOp.builder("deconv3d_bp")
                .addInputs(opInputs)
                .addIntegerArguments(args)
                .addOutputs(opOutputs)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);


        Gradient retGradient = new DefaultGradient();
        if(layerConf().hasBias()){
            retGradient.setGradientFor(DeconvolutionParamInitializer.BIAS_KEY, biasGradView);
        }
        retGradient.setGradientFor(DeconvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');
        weightNoiseParams.clear();

        return new Pair<>(retGradient, outEps);
    }

    protected INDArray preOutput(boolean training , LayerWorkspaceMgr workspaceMgr) {

        INDArray bias = getParamWithNoise(DeconvolutionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray weights = getParamWithNoise(DeconvolutionParamInitializer.WEIGHT_KEY, training, workspaceMgr);

        //Input validation: expect rank 5 matrix
        if (input.rank() != 5) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to Deconvolution3DLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 5 array with shape [minibatchSize, channels, inputHeight, inputWidth, inputDepth] or" +
                    " [minibatchSize, inputHeight, inputWidth, inputDepth, channels]. " + layerId());
        }

        Convolution3D.DataFormat df = layerConf().getDataFormat();
        boolean ncdhw = layerConf().getDataFormat() == Convolution3D.DataFormat.NCDHW;
        int chDim = ncdhw ? 1 : 4;
        if (input.size(chDim) != layerConf().getNIn() ) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Cannot do forward pass in Deconvolution3D layer (layer name = " + layerName
                    + ", layer index = " + index + "): input array channels does not match CNN layer configuration"
                    + " (data input channels = " + input.size(chDim) + ", " + (ncdhw ? "[minibatch,channels,height,width,depth]=" : "[minibatch,height,width,depth,channels]=")
                    + Arrays.toString(input.shape()) + "; expected" + " input channels = " + layerConf().getNIn() + ") "
                    + layerId());
        }

        int[] dilation = layerConf().getDilation();
        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();

        int[] pad;
        ConvolutionMode cm = layerConf().getConvolutionMode();
        long[] outSize;
        int[] inSize = df == Convolution3D.DataFormat.NCDHW ? new int[]{(int)input.size(2), (int)input.size(3), (int)input.size(4)} : new int[]{(int)input.size(1), (int)input.size(2), (int)input.size(3)};
        if (cm == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getDeconvolution3DOutputSize(input, kernel, strides, null, dilation, cm, layerConf().getDataFormat()); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(ArrayUtil.toInts(outSize), inSize, kernel, strides, dilation );
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getDeconvolution3DOutputSize(input, kernel, strides, pad, dilation, cm, layerConf().getDataFormat()); //Also performs validation
        }

        long outH = outSize[0];
        long outW = outSize[1];
        long outD = outSize[2];


        val miniBatch = input.size(0);
        long[] outShape = df == Convolution3D.DataFormat.NCDHW ? new long[]{miniBatch, layerConf().getNOut(), outH, outW, outD} : new long[]{miniBatch, outH, outW, outD, layerConf().getNOut()};
        INDArray output = workspaceMgr.create(ArrayType.ACTIVATIONS, input.dataType(), outShape, 'c');

        int sameMode = (cm == ConvolutionMode.Same) ? 1 : 0;

        int[] args = new int[] {
                kernel[0], kernel[1], kernel[2], strides[0], strides[1], strides[2],
                pad[0], pad[1], pad[2], dilation[0], dilation[1], dilation[2], sameMode,
                df == Convolution3D.DataFormat.NCDHW ? 0 : 1
        };

        INDArray[] opInputs;
        if (layerConf().hasBias()) {
            opInputs = new INDArray[]{input, weights, bias};
        } else {
            opInputs = new INDArray[]{input, weights};
        }
        CustomOp op = DynamicCustomOp.builder("deconv3d")
                .addInputs(opInputs)
                .addIntegerArguments(args)
                .addOutputs(output)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);

        return output;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        applyDropOutIfNecessary(training, workspaceMgr);

        INDArray z = preOutput(training, workspaceMgr);

        IActivation afn = layerConf().getActivationFn();

        INDArray activation = afn.getActivation(z, training);
        return activation;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
}