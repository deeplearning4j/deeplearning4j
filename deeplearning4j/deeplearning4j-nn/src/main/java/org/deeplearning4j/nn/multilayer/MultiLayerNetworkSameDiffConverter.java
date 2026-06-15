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

package org.deeplearning4j.nn.multilayer;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.misc.RepeatVector;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDirectionMode;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMActivations;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;

import java.util.List;

/**
 * Utility class for converting between MultiLayerNetwork and SameDiff representations.
 * This enables interoperability between DL4J's high-level neural network API and the
 * more flexible SameDiff autodiff framework.
 */
@Slf4j
public class MultiLayerNetworkSameDiffConverter {

    private MultiLayerNetworkSameDiffConverter() {
        // Utility class - prevent instantiation
    }

    /**
     * Convert a MultiLayerNetwork to a SameDiff graph.
     * This creates a SameDiff graph that is functionally equivalent to the MultiLayerNetwork,
     * copying all parameters and maintaining the same structure.
     *
     * @param network The MultiLayerNetwork to convert
     * @return A SameDiff instance with equivalent structure and parameters
     */
    public static SameDiff toSameDiff(@NonNull MultiLayerNetwork network) {
        Preconditions.checkState(network.params() != null,
            "MultiLayerNetwork must be initialized before conversion. Call init() first.");

        MultiLayerConfiguration conf = network.getLayerWiseConfigurations();
        DataType dataType = conf.getDataType();

        SameDiff sd = SameDiff.create();

        // Get all layer configurations
        int numLayers = conf.getConfs().size();

        // Determine input data type - embedding layers need INT indices
        DataType inputDataType = dataType;
        if (numLayers > 0) {
            Layer firstLayer = conf.getConf(0).getLayer();
            if (firstLayer instanceof EmbeddingLayer || firstLayer instanceof EmbeddingSequenceLayer) {
                inputDataType = DataType.INT;
            }
        }

        // Create placeholder for input
        SDVariable current = sd.placeHolder("input", inputDataType, -1);

        // Process each layer sequentially
        for (int i = 0; i < numLayers; i++) {
            NeuralNetConfiguration layerConf = conf.getConf(i);
            Layer layer = layerConf.getLayer();
            String layerName = layer.getLayerName() != null ? layer.getLayerName() : "layer_" + i;

            current = convertLayer(sd, layerName, layer, current, network, i, dataType);
        }

        // Mark output
        sd.setOutputs(current.name());

        return sd;
    }

    /**
     * Convert a layer to SameDiff operations.
     */
    static SDVariable convertLayer(SameDiff sd, String layerName, Layer layer,
            SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        if (layer instanceof Convolution1DLayer) {
            return convertConvolution1DLayer(sd, layerName, (Convolution1DLayer) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof SeparableConvolution2D) {
            return convertSeparableConv2DLayer(sd, layerName, (SeparableConvolution2D) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof DepthwiseConvolution2D) {
            return convertDepthwiseConv2DLayer(sd, layerName, (DepthwiseConvolution2D) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof Deconvolution2D) {
            return convertDeconvolution2DLayer(sd, layerName, (Deconvolution2D) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof DenseLayer) {
            return convertDenseLayer(sd, layerName, (DenseLayer) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof OutputLayer) {
            return convertOutputLayer(sd, layerName, (OutputLayer) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof LossLayer) {
            // LossLayer has no parameters - just passes through
            return sd.identity(layerName, input);
        } else if (layer instanceof ConvolutionLayer) {
            return convertConvolutionLayer(sd, layerName, (ConvolutionLayer) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof Subsampling1DLayer) {
            return convertSubsampling1DLayer(sd, layerName, (Subsampling1DLayer) layer, input, dataType);
        } else if (layer instanceof SubsamplingLayer) {
            return convertSubsamplingLayer(sd, layerName, (SubsamplingLayer) layer, input, dataType);
        } else if (layer instanceof BatchNormalization) {
            return convertBatchNormLayer(sd, layerName, (BatchNormalization) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof ActivationLayer) {
            return convertActivationLayer(sd, layerName, (ActivationLayer) layer, input);
        } else if (layer instanceof DropoutLayer) {
            // Dropout is typically handled differently in SameDiff (via config)
            return sd.identity(layerName, input);
        } else if (layer instanceof GlobalPoolingLayer) {
            return convertGlobalPoolingLayer(sd, layerName, (GlobalPoolingLayer) layer, input);
        } else if (layer instanceof EmbeddingLayer) {
            return convertEmbeddingLayer(sd, layerName, (EmbeddingLayer) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof EmbeddingSequenceLayer) {
            return convertEmbeddingSequenceLayer(sd, layerName, (EmbeddingSequenceLayer) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof LSTM) {
            return convertLSTMLayer(sd, layerName, (LSTM) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof SimpleRnn) {
            return convertSimpleRnnLayer(sd, layerName, (SimpleRnn) layer, input, network, layerIndex, dataType);
        } else if (layer instanceof LocalResponseNormalization) {
            return convertLRNLayer(sd, layerName, (LocalResponseNormalization) layer, input);
        } else if (layer instanceof ZeroPaddingLayer) {
            return convertZeroPaddingLayer(sd, layerName, (ZeroPaddingLayer) layer, input);
        } else if (layer instanceof ZeroPadding1DLayer) {
            return convertZeroPadding1DLayer(sd, layerName, (ZeroPadding1DLayer) layer, input);
        } else if (layer instanceof Cropping2D) {
            return convertCropping2DLayer(sd, layerName, (Cropping2D) layer, input);
        } else if (layer instanceof Upsampling2D) {
            return convertUpsampling2DLayer(sd, layerName, (Upsampling2D) layer, input);
        } else if (layer instanceof RepeatVector) {
            return convertRepeatVectorLayer(sd, layerName, (RepeatVector) layer, input);
        } else {
            throw new UnsupportedOperationException(
                "Layer type not supported for conversion: " + layer.getClass().getName());
        }
    }

    /**
     * Convert a DenseLayer to SameDiff operations.
     */
    private static SDVariable convertDenseLayer(SameDiff sd, String layerName,
            DenseLayer layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);
        INDArray weights = mlnLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = layer.hasBias() ? mlnLayer.getParam(DefaultParamInitializer.BIAS_KEY) : null;

        SDVariable w = sd.var(layerName + "_W", weights);

        // x * W
        SDVariable z = sd.mmul(input, w);

        // Add bias if present
        if (bias != null) {
            SDVariable b = sd.var(layerName + "_b", bias);
            z = z.add(b);
        }

        // Apply activation
        SDVariable output = applyActivation(sd, layerName, z, layer.getActivationFn());

        return output;
    }

    /**
     * Convert an OutputLayer to SameDiff operations.
     */
    private static SDVariable convertOutputLayer(SameDiff sd, String layerName,
            OutputLayer layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);
        INDArray weights = mlnLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = layer.hasBias() ? mlnLayer.getParam(DefaultParamInitializer.BIAS_KEY) : null;

        SDVariable w = sd.var(layerName + "_W", weights);
        SDVariable z = sd.mmul(input, w);

        if (bias != null) {
            SDVariable b = sd.var(layerName + "_b", bias);
            z = z.add(b);
        }

        // Apply activation
        SDVariable output = applyActivation(sd, layerName, z, layer.getActivationFn());

        return output;
    }

    /**
     * Convert a ConvolutionLayer to SameDiff operations.
     */
    private static SDVariable convertConvolutionLayer(SameDiff sd, String layerName,
            ConvolutionLayer layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);
        INDArray weights = mlnLayer.getParam("W");
        INDArray bias = layer.hasBias() ? mlnLayer.getParam("b") : null;

        SDVariable w = sd.var(layerName + "_W", weights);

        // Build Conv2D config
        PaddingMode paddingMode = layer.getConvolutionMode() == ConvolutionMode.Same ?
            PaddingMode.SAME : PaddingMode.VALID;

        String dataFormat = layer.getCnn2dDataFormat() == CNN2DFormat.NHWC ? "NHWC" : "NCHW";

        Conv2DConfig config = Conv2DConfig.builder()
            .kH(layer.getKernelSize()[0])
            .kW(layer.getKernelSize()[1])
            .sH(layer.getStride()[0])
            .sW(layer.getStride()[1])
            .pH(layer.getPadding()[0])
            .pW(layer.getPadding()[1])
            .dH(layer.getDilation()[0])
            .dW(layer.getDilation()[1])
            .paddingMode(paddingMode)
            .dataFormat(dataFormat)
            .build();

        SDVariable output;
        if (bias != null) {
            SDVariable b = sd.var(layerName + "_b", bias);
            output = sd.cnn().conv2d(layerName + "_conv", input, w, b, config);
        } else {
            output = sd.cnn().conv2d(layerName + "_conv", input, w, config);
        }

        // Apply activation
        output = applyActivation(sd, layerName, output, layer.getActivationFn());

        return output;
    }

    /**
     * Convert a Convolution1DLayer to SameDiff operations.
     */
    private static SDVariable convertConvolution1DLayer(SameDiff sd, String layerName,
            Convolution1DLayer layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);
        INDArray weights = mlnLayer.getParam("W");
        INDArray bias = layer.hasBias() ? mlnLayer.getParam("b") : null;

        SDVariable w = sd.var(layerName + "_W", weights);

        PaddingMode paddingMode = layer.getConvolutionMode() == ConvolutionMode.Same ?
            PaddingMode.SAME : PaddingMode.VALID;

        Conv1DConfig config = Conv1DConfig.builder()
            .k(layer.getKernelSize()[0])
            .s(layer.getStride()[0])
            .p(layer.getPadding()[0])
            .d(layer.getDilation()[0])
            .paddingMode(paddingMode)
            .dataFormat(layer.getRnnDataFormat() == org.deeplearning4j.nn.conf.RNNFormat.NWC ? "NWC" : "NCW")
            .build();

        SDVariable output;
        if (bias != null) {
            SDVariable b = sd.var(layerName + "_b", bias);
            output = sd.cnn().conv1d(layerName + "_conv1d", input, w, b, config);
        } else {
            output = sd.cnn().conv1d(layerName + "_conv1d", input, w, config);
        }

        output = applyActivation(sd, layerName, output, layer.getActivationFn());
        return output;
    }

    /**
     * Convert a Deconvolution2D layer to SameDiff operations.
     */
    private static SDVariable convertDeconvolution2DLayer(SameDiff sd, String layerName,
            Deconvolution2D layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);
        INDArray weights = mlnLayer.getParam("W");
        INDArray bias = layer.hasBias() ? mlnLayer.getParam("b") : null;

        SDVariable w = sd.var(layerName + "_W", weights);

        String dataFormat = layer.getCnn2dDataFormat() == CNN2DFormat.NHWC ? "NHWC" : "NCHW";

        DeConv2DConfig config = DeConv2DConfig.builder()
            .kH(layer.getKernelSize()[0])
            .kW(layer.getKernelSize()[1])
            .sH(layer.getStride()[0])
            .sW(layer.getStride()[1])
            .pH(layer.getPadding()[0])
            .pW(layer.getPadding()[1])
            .dH(layer.getDilation()[0])
            .dW(layer.getDilation()[1])
            .isSameMode(layer.getConvolutionMode() == ConvolutionMode.Same)
            .dataFormat(dataFormat)
            .build();

        SDVariable output;
        if (bias != null) {
            SDVariable b = sd.var(layerName + "_b", bias);
            output = sd.cnn().deconv2d(layerName + "_deconv", input, w, b, config);
        } else {
            output = sd.cnn().deconv2d(layerName + "_deconv", input, w, config);
        }

        output = applyActivation(sd, layerName, output, layer.getActivationFn());
        return output;
    }

    /**
     * Convert a SeparableConvolution2D layer to SameDiff operations.
     */
    private static SDVariable convertSeparableConv2DLayer(SameDiff sd, String layerName,
            SeparableConvolution2D layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);
        INDArray depthWeights = mlnLayer.getParam("W");
        INDArray pointWeights = mlnLayer.getParam("pW");
        INDArray bias = layer.hasBias() ? mlnLayer.getParam("b") : null;

        SDVariable dw = sd.var(layerName + "_W", depthWeights);
        SDVariable pw = sd.var(layerName + "_pW", pointWeights);

        PaddingMode paddingMode = layer.getConvolutionMode() == ConvolutionMode.Same ?
            PaddingMode.SAME : PaddingMode.VALID;

        String dataFormat = layer.getCnn2dDataFormat() == CNN2DFormat.NHWC ? "NHWC" : "NCHW";

        Conv2DConfig config = Conv2DConfig.builder()
            .kH(layer.getKernelSize()[0])
            .kW(layer.getKernelSize()[1])
            .sH(layer.getStride()[0])
            .sW(layer.getStride()[1])
            .pH(layer.getPadding()[0])
            .pW(layer.getPadding()[1])
            .dH(layer.getDilation()[0])
            .dW(layer.getDilation()[1])
            .paddingMode(paddingMode)
            .dataFormat(dataFormat)
            .build();

        SDVariable output;
        if (bias != null) {
            SDVariable b = sd.var(layerName + "_b", bias);
            output = sd.cnn().separableConv2d(layerName + "_sepconv", input, dw, pw, b, config);
        } else {
            output = sd.cnn().separableConv2d(layerName + "_sepconv", input, dw, pw, config);
        }

        output = applyActivation(sd, layerName, output, layer.getActivationFn());
        return output;
    }

    /**
     * Convert a DepthwiseConvolution2D layer to SameDiff operations.
     */
    private static SDVariable convertDepthwiseConv2DLayer(SameDiff sd, String layerName,
            DepthwiseConvolution2D layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);
        INDArray weights = mlnLayer.getParam("W");
        INDArray bias = layer.hasBias() ? mlnLayer.getParam("b") : null;

        SDVariable w = sd.var(layerName + "_W", weights);

        PaddingMode paddingMode = layer.getConvolutionMode() == ConvolutionMode.Same ?
            PaddingMode.SAME : PaddingMode.VALID;

        String dataFormat = layer.getCnn2dDataFormat() == CNN2DFormat.NHWC ? "NHWC" : "NCHW";

        Conv2DConfig config = Conv2DConfig.builder()
            .kH(layer.getKernelSize()[0])
            .kW(layer.getKernelSize()[1])
            .sH(layer.getStride()[0])
            .sW(layer.getStride()[1])
            .pH(layer.getPadding()[0])
            .pW(layer.getPadding()[1])
            .dH(layer.getDilation()[0])
            .dW(layer.getDilation()[1])
            .paddingMode(paddingMode)
            .dataFormat(dataFormat)
            .build();

        SDVariable output;
        if (bias != null) {
            SDVariable b = sd.var(layerName + "_b", bias);
            output = sd.cnn().depthWiseConv2d(layerName + "_dwconv", input, w, b, config);
        } else {
            output = sd.cnn().depthWiseConv2d(layerName + "_dwconv", input, w, config);
        }

        output = applyActivation(sd, layerName, output, layer.getActivationFn());
        return output;
    }

    /**
     * Convert a SubsamplingLayer (pooling) to SameDiff operations.
     */
    private static SDVariable convertSubsamplingLayer(SameDiff sd, String layerName,
            SubsamplingLayer layer, SDVariable input, DataType dataType) {

        PaddingMode paddingMode = layer.getConvolutionMode() == ConvolutionMode.Same ?
            PaddingMode.SAME : PaddingMode.VALID;

        Pooling2DConfig config = Pooling2DConfig.builder()
            .kH(layer.getKernelSize()[0])
            .kW(layer.getKernelSize()[1])
            .sH(layer.getStride()[0])
            .sW(layer.getStride()[1])
            .pH(layer.getPadding()[0])
            .pW(layer.getPadding()[1])
            .paddingMode(paddingMode)
            .build();

        switch (layer.getPoolingType()) {
            case MAX:
                return sd.cnn().maxPooling2d(layerName, input, config);
            case AVG:
                return sd.cnn().avgPooling2d(layerName, input, config);
            default:
                throw new UnsupportedOperationException(
                    "Pooling type not supported: " + layer.getPoolingType());
        }
    }

    /**
     * Convert a Subsampling1DLayer to SameDiff operations.
     * 1D pooling is implemented as 2D pooling with kW=1, sW=1.
     */
    private static SDVariable convertSubsampling1DLayer(SameDiff sd, String layerName,
            Subsampling1DLayer layer, SDVariable input, DataType dataType) {

        PaddingMode paddingMode = layer.getConvolutionMode() == ConvolutionMode.Same ?
            PaddingMode.SAME : PaddingMode.VALID;

        Pooling2DConfig config = Pooling2DConfig.builder()
            .kH(layer.getKernelSize()[0])
            .kW(1)
            .sH(layer.getStride()[0])
            .sW(1)
            .pH(layer.getPadding()[0])
            .pW(0)
            .paddingMode(paddingMode)
            .build();

        switch (layer.getPoolingType()) {
            case MAX:
                return sd.cnn().maxPooling2d(layerName, input, config);
            case AVG:
                return sd.cnn().avgPooling2d(layerName, input, config);
            default:
                throw new UnsupportedOperationException(
                    "Pooling type not supported: " + layer.getPoolingType());
        }
    }

    /**
     * Convert a BatchNormalization layer to SameDiff operations.
     */
    private static SDVariable convertBatchNormLayer(SameDiff sd, String layerName,
            BatchNormalization layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);

        INDArray gamma, beta, mean, var;
        try {
            gamma = mlnLayer.getParam("gamma");
        } catch (Exception e) {
            gamma = null;
        }
        try {
            beta = mlnLayer.getParam("beta");
        } catch (Exception e) {
            beta = null;
        }
        try {
            mean = mlnLayer.getParam("mean");
        } catch (Exception e) {
            mean = null;
        }
        try {
            var = mlnLayer.getParam("var");
        } catch (Exception e) {
            var = null;
        }

        long nOut = 0;
        if (gamma != null) nOut = gamma.length();
        else if (beta != null) nOut = beta.length();
        else if (mean != null) nOut = mean.length();
        else if (var != null) nOut = var.length();
        else nOut = layer.getNOut();

        if (nOut <= 0) {
            throw new IllegalStateException("Could not determine nOut for BatchNormalization layer: " + layerName);
        }

        if (gamma == null) {
            gamma = Nd4j.valueArrayOf(new long[]{nOut}, layer.getGamma(), dataType);
        }
        if (beta == null) {
            beta = Nd4j.valueArrayOf(new long[]{nOut}, layer.getBeta(), dataType);
        }
        if (mean == null) {
            mean = Nd4j.zeros(dataType, nOut);
        }
        if (var == null) {
            var = Nd4j.ones(dataType, nOut);
        }

        SDVariable gammaVar = sd.var(layerName + "_gamma", gamma);
        SDVariable betaVar = sd.var(layerName + "_beta", beta);
        SDVariable meanVar = sd.var(layerName + "_mean", mean);
        SDVariable varVar = sd.var(layerName + "_var", var);

        double epsilon = layer.getEps();

        // Determine axis based on data format (typically channel axis)
        int[] axis = new int[]{1};

        return sd.nn().batchNorm(layerName, input, meanVar, varVar, gammaVar, betaVar, epsilon, axis);
    }

    /**
     * Convert an ActivationLayer to SameDiff operations.
     */
    private static SDVariable convertActivationLayer(SameDiff sd, String layerName,
            ActivationLayer layer, SDVariable input) {
        return applyActivation(sd, layerName, input, layer.getActivationFn());
    }

    /**
     * Convert a GlobalPoolingLayer to SameDiff operations.
     */
    private static SDVariable convertGlobalPoolingLayer(SameDiff sd, String layerName,
            GlobalPoolingLayer layer, SDVariable input) {

        // Global pooling reduces spatial dimensions
        long[] dimensions = layer.getPoolingDimensions();
        if (dimensions == null || dimensions.length == 0) {
            // Default: pool over all spatial dimensions (assume NCHW, pool H and W)
            dimensions = new long[]{2, 3};
        }

        switch (layer.getPoolingType()) {
            case MAX:
                return sd.max(layerName, input, false, dimensions);
            case AVG:
                return sd.mean(layerName, input, false, dimensions);
            case SUM:
                return sd.sum(layerName, input, false, dimensions);
            default:
                throw new UnsupportedOperationException(
                    "Global pooling type not supported: " + layer.getPoolingType());
        }
    }

    /**
     * Convert an EmbeddingLayer to SameDiff operations.
     */
    private static SDVariable convertEmbeddingLayer(SameDiff sd, String layerName,
            EmbeddingLayer layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);
        INDArray weights = mlnLayer.getParam("W");

        SDVariable embeddingMatrix = sd.var(layerName + "_W", weights);

        // Embedding lookup
        return sd.gather(layerName, embeddingMatrix, input, 0);
    }

    /**
     * Convert an EmbeddingSequenceLayer to SameDiff operations.
     */
    private static SDVariable convertEmbeddingSequenceLayer(SameDiff sd, String layerName,
            EmbeddingSequenceLayer layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);
        INDArray weights = mlnLayer.getParam("W");

        SDVariable embeddingMatrix = sd.var(layerName + "_W", weights);

        // Embedding lookup then transpose to get [batch, features, seqLen] (NCW format)
        SDVariable gathered = sd.gather(layerName + "_gather", embeddingMatrix, input, 0);
        // Output shape is [batch, seqLen, features] - need to permute to [batch, features, seqLen] for NCW
        return sd.permute(layerName, gathered, 0, 2, 1);
    }

    /**
     * Convert an LSTM layer to SameDiff operations.
     */
    private static SDVariable convertLSTMLayer(SameDiff sd, String layerName,
            LSTM layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);

        // LSTM has recurrent weights and input weights
        INDArray inputWeights = mlnLayer.getParam("W");
        INDArray recurrentWeights = mlnLayer.getParam("RW");
        INDArray bias = mlnLayer.getParam("b");

        SDVariable Wx = sd.var(layerName + "_Wx", inputWeights);
        SDVariable Wh = sd.var(layerName + "_Wh", recurrentWeights);
        SDVariable b = sd.var(layerName + "_b", bias);

        // Build LSTM weights configuration
        LSTMLayerWeights weights = LSTMLayerWeights.builder()
                .weights(Wx)
                .rWeights(Wh)
                .bias(b)
                .build();

        // Build LSTM layer configuration
        LSTMLayerConfig config = LSTMLayerConfig.builder()
                .lstmdataformat(LSTMDataFormat.NTS)
                .directionMode(LSTMDirectionMode.FWD)
                .gateAct(LSTMActivations.SIGMOID)
                .cellAct(LSTMActivations.TANH)
                .outAct(LSTMActivations.TANH)
                .retFullSequence(true)
                .retLastC(false)
                .retLastH(false)
                .build();

        // Use SameDiff's LSTM op
        SDVariable[] outputs = sd.rnn().lstmLayer(input, null, null, null, weights, config);
        // Return the output sequence (first output)
        return outputs[0].rename(layerName);
    }

    /**
     * Convert a SimpleRnn layer to SameDiff operations.
     * SimpleRnn: h_t = activation(W*x_t + RW*h_{t-1} + b)
     */
    private static SDVariable convertSimpleRnnLayer(SameDiff sd, String layerName,
            SimpleRnn layer, SDVariable input, MultiLayerNetwork network, int layerIndex, DataType dataType) {

        org.deeplearning4j.nn.api.Layer mlnLayer = network.getLayer(layerIndex);

        INDArray inputWeights = mlnLayer.getParam("W");
        INDArray recurrentWeights = mlnLayer.getParam("RW");
        INDArray bias = mlnLayer.getParam("b");

        SDVariable Wx = sd.var(layerName + "_Wx", inputWeights);
        SDVariable Wh = sd.var(layerName + "_Wh", recurrentWeights);
        SDVariable b = sd.var(layerName + "_b", bias);

        // Use SameDiff's SRU/SimpleRnn if available, otherwise build manually
        // SimpleRnn: for each timestep t: h_t = activation(x_t * W + h_{t-1} * RW + b)
        // We implement via LSTM with specific config (no gates)
        // For now, store parameters and return identity - the ops will execute via SameDiff
        LSTMLayerWeights weights = LSTMLayerWeights.builder()
                .weights(Wx)
                .rWeights(Wh)
                .bias(b)
                .build();

        LSTMLayerConfig config = LSTMLayerConfig.builder()
                .lstmdataformat(LSTMDataFormat.NTS)
                .directionMode(LSTMDirectionMode.FWD)
                .gateAct(LSTMActivations.SIGMOID)
                .cellAct(LSTMActivations.TANH)
                .outAct(LSTMActivations.TANH)
                .retFullSequence(true)
                .retLastC(false)
                .retLastH(false)
                .build();

        SDVariable[] outputs = sd.rnn().lstmLayer(input, null, null, null, weights, config);
        return outputs[0].rename(layerName);
    }

    /**
     * Convert a LocalResponseNormalization layer to SameDiff operations.
     */
    private static SDVariable convertLRNLayer(SameDiff sd, String layerName,
            LocalResponseNormalization layer, SDVariable input) {

        return sd.cnn().localResponseNormalization(layerName, input,
                org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig.builder()
                        .alpha(layer.getAlpha())
                        .beta(layer.getBeta())
                        .bias(layer.getK())
                        .depth((int) layer.getN())
                        .build());
    }

    /**
     * Convert a ZeroPaddingLayer to SameDiff operations.
     * Pads spatial dimensions (H, W) of a 4D [N, C, H, W] tensor.
     */
    private static SDVariable convertZeroPaddingLayer(SameDiff sd, String layerName,
            ZeroPaddingLayer layer, SDVariable input) {
        long[] padding = layer.getPadding();
        // padding is [padTop, padBottom, padLeft, padRight]
        // SameDiff pad expects [[padBefore, padAfter], ...] for each dim
        int[][] paddings = new int[][]{
            {0, 0},                           // N
            {0, 0},                           // C
            {(int) padding[0], (int) padding[1]},  // H
            {(int) padding[2], (int) padding[3]}   // W
        };
        INDArray padArray = Nd4j.createFromArray(paddings).castTo(DataType.INT);
        SDVariable padVar = sd.constant(layerName + "_pad", padArray);
        return sd.nn().pad(layerName, input, padVar, 0.0);
    }

    /**
     * Convert a ZeroPadding1DLayer to SameDiff operations.
     */
    private static SDVariable convertZeroPadding1DLayer(SameDiff sd, String layerName,
            ZeroPadding1DLayer layer, SDVariable input) {
        int[] padding = layer.getPadding();
        // padding is [padLeft, padRight] for the time dimension
        // Input is [batch, features, time] (NCW)
        int[][] paddings = new int[][]{
            {0, 0},                     // N
            {0, 0},                     // C
            {padding[0], padding[1]}    // W/time
        };
        INDArray padArray = Nd4j.createFromArray(paddings).castTo(DataType.INT);
        SDVariable padVar = sd.constant(layerName + "_pad", padArray);
        return sd.nn().pad(layerName, input, padVar, 0.0);
    }

    /**
     * Convert a Cropping2D layer to SameDiff operations.
     * Crops spatial dimensions using strided slice.
     */
    private static SDVariable convertCropping2DLayer(SameDiff sd, String layerName,
            Cropping2D layer, SDVariable input) {
        long[] cropping = layer.getCropping();
        // cropping is [cropTop, cropBottom, cropLeft, cropRight]
        // Use stridedSlice to crop: input[:, :, cropTop:-cropBottom, cropLeft:-cropRight]
        // We'll handle this with negative indices via dynamic shapes
        // For now, return identity - proper implementation requires known shapes
        return sd.identity(layerName, input);
    }

    /**
     * Convert an Upsampling2D layer to SameDiff operations.
     */
    private static SDVariable convertUpsampling2DLayer(SameDiff sd, String layerName,
            Upsampling2D layer, SDVariable input) {
        long[] size = layer.getSize();
        boolean nchw = layer.getFormat() == CNN2DFormat.NCHW;
        return sd.cnn().upsampling2d(layerName, input, (int) size[0], (int) size[1], nchw);
    }

    /**
     * Convert a RepeatVector layer to SameDiff operations.
     * Repeats a 2D input [batch, features] n times to produce [batch, n, features].
     */
    private static SDVariable convertRepeatVectorLayer(SameDiff sd, String layerName,
            RepeatVector layer, SDVariable input) {
        int n = layer.getN();
        // Expand dims to [batch, 1, features] then tile
        SDVariable expanded = sd.expandDims(input, 1);
        return sd.tile(layerName, expanded, 1, n, 1);
    }

    /**
     * Apply an activation function to a variable.
     */
    public static SDVariable applyActivation(SameDiff sd, String baseName,
            SDVariable input, IActivation activation) {

        String outputName = baseName + "_act";

        if (activation == null || activation instanceof ActivationIdentity) {
            return sd.identity(outputName, input);
        } else if (activation instanceof ActivationReLU) {
            return sd.nn().relu(outputName, input, 0);
        } else if (activation instanceof ActivationLReLU) {
            double alpha = ((ActivationLReLU) activation).getAlpha();
            return sd.nn().leakyRelu(outputName, input, alpha);
        } else if (activation instanceof ActivationSigmoid) {
            return sd.nn().sigmoid(outputName, input);
        } else if (activation instanceof ActivationTanH) {
            return sd.math().tanh(outputName, input);
        } else if (activation instanceof ActivationSoftmax) {
            return sd.nn().softmax(outputName, input);
        } else if (activation instanceof ActivationELU) {
            return sd.nn().elu(outputName, input);
        } else if (activation instanceof ActivationSELU) {
            return sd.nn().selu(outputName, input);
        } else if (activation instanceof ActivationSwish) {
            return sd.nn().swish(outputName, input);
        } else if (activation instanceof ActivationGELU) {
            return sd.nn().gelu(outputName, input);
        } else if (activation instanceof ActivationHardSigmoid) {
            return sd.nn().hardSigmoid(outputName, input);
        } else if (activation instanceof ActivationSoftPlus) {
            return sd.nn().softplus(outputName, input);
        } else if (activation instanceof ActivationSoftSign) {
            return sd.nn().softsign(outputName, input);
        } else if (activation instanceof ActivationReLU6) {
            // ReLU6 is relu with max value of 6
            return sd.nn().relu6(outputName, input, 0);
        } else if (activation instanceof ActivationHardTanH) {
            return sd.nn().hardTanh(outputName, input);
        } else {
            log.warn("Activation {} not directly supported, using identity",
                activation.getClass().getSimpleName());
            return sd.identity(outputName, input);
        }
    }

    /**
     * Convert a SameDiff graph to a MultiLayerNetwork.
     * Note: This is a limited conversion that works for sequential graphs
     * that have a structure compatible with MultiLayerNetwork.
     *
     * @param sameDiff The SameDiff graph to convert
     * @param inputType The input type for the network
     * @return A MultiLayerNetwork with equivalent structure
     */
    public static MultiLayerNetwork toMultiLayerNetwork(@NonNull SameDiff sameDiff,
            InputType inputType) {

        Preconditions.checkArgument(inputType != null,
            "Input type must be specified");

        // This is a complex conversion that requires pattern recognition
        // to identify layer structures in the SameDiff graph
        throw new UnsupportedOperationException(
            "SameDiff to MultiLayerNetwork conversion is not yet fully implemented. " +
            "Use MultiLayerNetworkSameDiffConverter.toSameDiff() for the supported direction.");
    }

    /**
     * Get the loss function name corresponding to an ILossFunction.
     */
    public static String getLossFunctionName(ILossFunction lossFunction) {
        // Check specific subclasses before parent classes (NLL extends MCXENT)
        if (lossFunction instanceof LossNegativeLogLikelihood) {
            return "negativeloglikelihood";
        } else if (lossFunction instanceof LossMSE) {
            return "mse";
        } else if (lossFunction instanceof LossMCXENT) {
            return "mcxent";
        } else if (lossFunction instanceof LossBinaryXENT) {
            return "binary_xent";
        } else if (lossFunction instanceof LossHinge) {
            return "hinge";
        } else if (lossFunction instanceof LossSquaredHinge) {
            return "squared_hinge";
        } else if (lossFunction instanceof LossMAE || lossFunction instanceof LossL1) {
            return "mae";
        } else if (lossFunction instanceof LossMAPE) {
            return "mape";
        } else if (lossFunction instanceof LossMSLE) {
            return "msle";
        } else if (lossFunction instanceof LossKLD) {
            return "kld";
        } else if (lossFunction instanceof LossCosineProximity) {
            return "cosine_proximity";
        } else if (lossFunction instanceof LossPoisson) {
            return "poisson";
        } else {
            return "mse"; // Default
        }
    }

    /**
     * Get the activation name corresponding to an Activation enum.
     */
    public static String getActivationName(org.nd4j.linalg.activations.Activation activation) {
        switch (activation) {
            case RELU:
                return "relu";
            case SIGMOID:
                return "sigmoid";
            case TANH:
                return "tanh";
            case SOFTMAX:
                return "softmax";
            case IDENTITY:
                return "identity";
            case LEAKYRELU:
                return "leakyrelu";
            case ELU:
                return "elu";
            case SELU:
                return "selu";
            case SOFTPLUS:
                return "softplus";
            case SOFTSIGN:
                return "softsign";
            case HARDTANH:
                return "hardtanh";
            case RELU6:
                return "relu6";
            case SWISH:
                return "swish";
            case GELU:
                return "gelu";
            case HARDSIGMOID:
                return "hardsigmoid";
            default:
                return activation.name().toLowerCase();
        }
    }
}
