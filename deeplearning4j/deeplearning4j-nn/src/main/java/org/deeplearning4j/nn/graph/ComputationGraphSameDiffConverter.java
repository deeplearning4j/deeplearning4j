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

package org.deeplearning4j.nn.graph;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.misc.RepeatVector;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.multilayer.MultiLayerNetworkSameDiffConverter;
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

import java.util.*;

/**
 * Utility class for converting between ComputationGraph and SameDiff representations.
 * This enables interoperability between DL4J's high-level neural network API and the
 * more flexible SameDiff autodiff framework.
 */
@Slf4j
public class ComputationGraphSameDiffConverter {

    private ComputationGraphSameDiffConverter() {
        // Utility class - prevent instantiation
    }

    /**
     * Convert a ComputationGraph to a SameDiff graph.
     * This creates a SameDiff graph that is functionally equivalent to the ComputationGraph,
     * copying all parameters and maintaining the same structure.
     *
     * @param computationGraph The ComputationGraph to convert
     * @return A SameDiff instance with equivalent structure and parameters
     */
    public static SameDiff toSameDiff(@NonNull ComputationGraph computationGraph) {
        // Check if initialized by checking if params exist
        Preconditions.checkState(computationGraph.params() != null,
            "ComputationGraph must be initialized before conversion. Call init() first.");

        ComputationGraphConfiguration config = computationGraph.getConfiguration();
        DataType dataType = config.getDataType();

        SameDiff sd = SameDiff.create();

        // Map to track vertex name -> SDVariable output
        Map<String, SDVariable> vertexOutputs = new LinkedHashMap<>();

        // Get topological order
        List<String> topologicalOrder = config.getTopologicalOrderStr();
        if (topologicalOrder == null) {
            topologicalOrder = new ArrayList<>(config.getVertices().keySet());
        }

        // Create placeholders for inputs
        List<String> networkInputs = config.getNetworkInputs();
        for (String inputName : networkInputs) {
            // Create placeholder - shape will be determined at runtime
            SDVariable placeholder = sd.placeHolder(inputName, dataType, -1);
            vertexOutputs.put(inputName, placeholder);
        }

        // Process each vertex in topological order
        for (String vertexName : topologicalOrder) {
            if (networkInputs.contains(vertexName)) {
                continue; // Skip input vertices
            }

            GraphVertex vertexConfig = config.getVertices().get(vertexName);
            List<String> inputNames = config.getVertexInputs().get(vertexName);

            // Gather input variables
            SDVariable[] inputs = new SDVariable[inputNames.size()];
            for (int i = 0; i < inputNames.size(); i++) {
                inputs[i] = vertexOutputs.get(inputNames.get(i));
                Preconditions.checkNotNull(inputs[i],
                    "Input '%s' for vertex '%s' not found", inputNames.get(i), vertexName);
            }

            // Convert vertex
            SDVariable output = convertVertex(sd, vertexName, vertexConfig, inputs,
                computationGraph, dataType);
            vertexOutputs.put(vertexName, output);
        }

        // Mark outputs - collect all output names and set them at once
        List<String> networkOutputs = config.getNetworkOutputs();
        List<String> outputVarNames = new ArrayList<>();
        for (String outputName : networkOutputs) {
            SDVariable outputVar = vertexOutputs.get(outputName);
            if (outputVar != null) {
                outputVarNames.add(outputVar.name());
            }
        }
        if (!outputVarNames.isEmpty()) {
            sd.setOutputs(outputVarNames.toArray(new String[0]));
        }

        return sd;
    }

    /**
     * Convert a single vertex to SameDiff operations.
     */
    private static SDVariable convertVertex(SameDiff sd, String vertexName,
            GraphVertex vertexConfig, SDVariable[] inputs,
            ComputationGraph cg, DataType dataType) {

        if (vertexConfig instanceof LayerVertex) {
            LayerVertex layerVertex = (LayerVertex) vertexConfig;
            Layer layer = layerVertex.getLayerConf().getLayer();
            return convertLayer(sd, vertexName, layer, inputs[0], cg, dataType);
        } else if (vertexConfig instanceof MergeVertex) {
            // Concatenate inputs along the channel/feature dimension
            return sd.concat(vertexName, 1, inputs);
        } else if (vertexConfig instanceof ElementWiseVertex) {
            ElementWiseVertex ewv = (ElementWiseVertex) vertexConfig;
            switch (ewv.getOp()) {
                case Add:
                    SDVariable result = inputs[0];
                    for (int i = 1; i < inputs.length; i++) {
                        result = result.add(inputs[i]);
                    }
                    return sd.updateVariableNameAndReference(result, vertexName);
                case Subtract:
                    return sd.updateVariableNameAndReference(
                        inputs[0].sub(inputs[1]), vertexName);
                case Product:
                    result = inputs[0];
                    for (int i = 1; i < inputs.length; i++) {
                        result = result.mul(inputs[i]);
                    }
                    return sd.updateVariableNameAndReference(result, vertexName);
                case Average:
                    return sd.mean(vertexName, sd.stack(0, inputs), false, 0);
                case Max:
                    return sd.max(vertexName, sd.stack(0, inputs), false, 0);
                default:
                    throw new UnsupportedOperationException(
                        "ElementWise operation not supported: " + ewv.getOp());
            }
        } else {
            throw new UnsupportedOperationException(
                "Vertex type not supported for conversion: " + vertexConfig.getClass().getName());
        }
    }

    /**
     * Convert a layer to SameDiff operations.
     * Delegates to layer-specific converters based on type.
     */
    private static SDVariable convertLayer(SameDiff sd, String layerName, Layer layer,
            SDVariable input, ComputationGraph cg, DataType dataType) {

        // Check subclasses before parents (Conv1D before Conv, etc.)
        if (layer instanceof Convolution1DLayer) {
            return convertConvolution1DLayer(sd, layerName, (Convolution1DLayer) layer, input, cg, dataType);
        } else if (layer instanceof SeparableConvolution2D) {
            return convertSeparableConv2DLayer(sd, layerName, (SeparableConvolution2D) layer, input, cg, dataType);
        } else if (layer instanceof DepthwiseConvolution2D) {
            return convertDepthwiseConv2DLayer(sd, layerName, (DepthwiseConvolution2D) layer, input, cg, dataType);
        } else if (layer instanceof Deconvolution2D) {
            return convertDeconvolution2DLayer(sd, layerName, (Deconvolution2D) layer, input, cg, dataType);
        } else if (layer instanceof DenseLayer) {
            return convertDenseLayer(sd, layerName, (DenseLayer) layer, input, cg, dataType);
        } else if (layer instanceof OutputLayer) {
            return convertOutputLayer(sd, layerName, (OutputLayer) layer, input, cg, dataType);
        } else if (layer instanceof LossLayer) {
            return sd.identity(layerName, input);
        } else if (layer instanceof ConvolutionLayer) {
            return convertConvolutionLayer(sd, layerName, (ConvolutionLayer) layer, input, cg, dataType);
        } else if (layer instanceof Subsampling1DLayer) {
            return convertSubsampling1DLayer(sd, layerName, (Subsampling1DLayer) layer, input, dataType);
        } else if (layer instanceof SubsamplingLayer) {
            return convertSubsamplingLayer(sd, layerName, (SubsamplingLayer) layer, input, dataType);
        } else if (layer instanceof BatchNormalization) {
            return convertBatchNormLayer(sd, layerName, (BatchNormalization) layer, input, cg, dataType);
        } else if (layer instanceof ActivationLayer) {
            return convertActivationLayer(sd, layerName, (ActivationLayer) layer, input);
        } else if (layer instanceof DropoutLayer) {
            return sd.identity(layerName, input);
        } else if (layer instanceof GlobalPoolingLayer) {
            return convertGlobalPoolingLayer(sd, layerName, (GlobalPoolingLayer) layer, input);
        } else if (layer instanceof EmbeddingLayer) {
            return convertEmbeddingLayer(sd, layerName, (EmbeddingLayer) layer, input, cg, dataType);
        } else if (layer instanceof EmbeddingSequenceLayer) {
            return convertEmbeddingSequenceLayer(sd, layerName, (EmbeddingSequenceLayer) layer, input, cg, dataType);
        } else if (layer instanceof LSTM) {
            return convertLSTMLayer(sd, layerName, (LSTM) layer, input, cg, dataType);
        } else if (layer instanceof SimpleRnn) {
            return convertSimpleRnnLayer(sd, layerName, (SimpleRnn) layer, input, cg, dataType);
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
            DenseLayer layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = layer.hasBias() ? cgLayer.getParam(DefaultParamInitializer.BIAS_KEY) : null;

        SDVariable w = sd.var(layerName + "_W", weights);
        SDVariable z = sd.mmul(input, w);

        if (bias != null) {
            SDVariable b = sd.var(layerName + "_b", bias);
            z = z.add(b);
        }

        return applyActivation(sd, layerName, z, layer.getActivationFn());
    }

    /**
     * Convert an OutputLayer to SameDiff operations.
     */
    private static SDVariable convertOutputLayer(SameDiff sd, String layerName,
            OutputLayer layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = layer.hasBias() ? cgLayer.getParam(DefaultParamInitializer.BIAS_KEY) : null;

        SDVariable w = sd.var(layerName + "_W", weights);
        SDVariable z = sd.mmul(input, w);

        if (bias != null) {
            SDVariable b = sd.var(layerName + "_b", bias);
            z = z.add(b);
        }

        return applyActivation(sd, layerName, z, layer.getActivationFn());
    }

    /**
     * Convert a ConvolutionLayer to SameDiff operations.
     */
    private static SDVariable convertConvolutionLayer(SameDiff sd, String layerName,
            ConvolutionLayer layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam("W");
        INDArray bias = layer.hasBias() ? cgLayer.getParam("b") : null;

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
            output = sd.cnn().conv2d(layerName + "_conv", input, w, b, config);
        } else {
            output = sd.cnn().conv2d(layerName + "_conv", input, w, config);
        }

        return applyActivation(sd, layerName, output, layer.getActivationFn());
    }

    /**
     * Convert a Convolution1DLayer to SameDiff operations.
     */
    private static SDVariable convertConvolution1DLayer(SameDiff sd, String layerName,
            Convolution1DLayer layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam("W");
        INDArray bias = layer.hasBias() ? cgLayer.getParam("b") : null;

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

        return applyActivation(sd, layerName, output, layer.getActivationFn());
    }

    /**
     * Convert a Deconvolution2D layer to SameDiff operations.
     */
    private static SDVariable convertDeconvolution2DLayer(SameDiff sd, String layerName,
            Deconvolution2D layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam("W");
        INDArray bias = layer.hasBias() ? cgLayer.getParam("b") : null;

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

        return applyActivation(sd, layerName, output, layer.getActivationFn());
    }

    /**
     * Convert a SeparableConvolution2D layer to SameDiff operations.
     */
    private static SDVariable convertSeparableConv2DLayer(SameDiff sd, String layerName,
            SeparableConvolution2D layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray depthWeights = cgLayer.getParam("W");
        INDArray pointWeights = cgLayer.getParam("pW");
        INDArray bias = layer.hasBias() ? cgLayer.getParam("b") : null;

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

        return applyActivation(sd, layerName, output, layer.getActivationFn());
    }

    /**
     * Convert a DepthwiseConvolution2D layer to SameDiff operations.
     */
    private static SDVariable convertDepthwiseConv2DLayer(SameDiff sd, String layerName,
            DepthwiseConvolution2D layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam("W");
        INDArray bias = layer.hasBias() ? cgLayer.getParam("b") : null;

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

        return applyActivation(sd, layerName, output, layer.getActivationFn());
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
            BatchNormalization layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);

        INDArray gamma, beta, mean, var;
        try {
            gamma = cgLayer.getParam("gamma");
        } catch (Exception e) {
            gamma = null;
        }
        try {
            beta = cgLayer.getParam("beta");
        } catch (Exception e) {
            beta = null;
        }
        try {
            mean = cgLayer.getParam("mean");
        } catch (Exception e) {
            mean = null;
        }
        try {
            var = cgLayer.getParam("var");
        } catch (Exception e) {
            var = null;
        }

        if (var == null) {
            try {
                var = cgLayer.getParam("log10stdev");
                if (var != null) {
                    var = Nd4j.math().pow(Nd4j.math().pow(Nd4j.scalar(10.0), var), 2);
                }
            } catch (Exception e) {
                // ignore
            }
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

        long[] dimensions = layer.getPoolingDimensions();
        if (dimensions == null || dimensions.length == 0) {
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
            EmbeddingLayer layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam("W");

        SDVariable embeddingMatrix = sd.var(layerName + "_W", weights);
        return sd.gather(layerName, embeddingMatrix, input, 0);
    }

    /**
     * Convert an EmbeddingSequenceLayer to SameDiff operations.
     */
    private static SDVariable convertEmbeddingSequenceLayer(SameDiff sd, String layerName,
            EmbeddingSequenceLayer layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam("W");

        SDVariable embeddingMatrix = sd.var(layerName + "_W", weights);
        SDVariable gathered = sd.gather(layerName + "_gather", embeddingMatrix, input, 0);
        return sd.permute(layerName, gathered, 0, 2, 1);
    }

    /**
     * Convert an LSTM layer to SameDiff operations.
     */
    private static SDVariable convertLSTMLayer(SameDiff sd, String layerName,
            LSTM layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);

        INDArray inputWeights = cgLayer.getParam("W");
        INDArray recurrentWeights = cgLayer.getParam("RW");
        INDArray bias = cgLayer.getParam("b");

        SDVariable Wx = sd.var(layerName + "_Wx", inputWeights);
        SDVariable Wh = sd.var(layerName + "_Wh", recurrentWeights);
        SDVariable b = sd.var(layerName + "_b", bias);

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
     * Convert a SimpleRnn layer to SameDiff operations.
     */
    private static SDVariable convertSimpleRnnLayer(SameDiff sd, String layerName,
            SimpleRnn layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);

        INDArray inputWeights = cgLayer.getParam("W");
        INDArray recurrentWeights = cgLayer.getParam("RW");
        INDArray bias = cgLayer.getParam("b");

        SDVariable Wx = sd.var(layerName + "_Wx", inputWeights);
        SDVariable Wh = sd.var(layerName + "_Wh", recurrentWeights);
        SDVariable b = sd.var(layerName + "_b", bias);

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
     */
    private static SDVariable convertZeroPaddingLayer(SameDiff sd, String layerName,
            ZeroPaddingLayer layer, SDVariable input) {
        long[] padding = layer.getPadding();
        int[][] paddings = new int[][]{
            {0, 0},
            {0, 0},
            {(int) padding[0], (int) padding[1]},
            {(int) padding[2], (int) padding[3]}
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
        int[][] paddings = new int[][]{
            {0, 0},
            {0, 0},
            {padding[0], padding[1]}
        };
        INDArray padArray = Nd4j.createFromArray(paddings).castTo(DataType.INT);
        SDVariable padVar = sd.constant(layerName + "_pad", padArray);
        return sd.nn().pad(layerName, input, padVar, 0.0);
    }

    /**
     * Convert a Cropping2D layer to SameDiff operations.
     */
    private static SDVariable convertCropping2DLayer(SameDiff sd, String layerName,
            Cropping2D layer, SDVariable input) {
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
     */
    private static SDVariable convertRepeatVectorLayer(SameDiff sd, String layerName,
            RepeatVector layer, SDVariable input) {
        int n = layer.getN();
        SDVariable expanded = sd.expandDims(input, 1);
        return sd.tile(layerName, expanded, 1, n, 1);
    }

    /**
     * Apply an activation function to a variable.
     */
    private static SDVariable applyActivation(SameDiff sd, String baseName,
            SDVariable input, IActivation activation) {
        // Delegate to shared implementation
        return MultiLayerNetworkSameDiffConverter.applyActivation(sd, baseName, input, activation);
    }

    /**
     * Convert a SameDiff graph to a ComputationGraph.
     */
    public static ComputationGraph toComputationGraph(@NonNull SameDiff sameDiff,
            InputType... inputTypes) {

        Preconditions.checkArgument(inputTypes != null && inputTypes.length > 0,
            "At least one input type must be specified");

        throw new UnsupportedOperationException(
            "SameDiff to ComputationGraph conversion is not yet fully implemented. " +
            "Use ComputationGraph.toSameDiff() for the supported direction.");
    }

    /**
     * Get the loss function corresponding to an ILossFunction.
     */
    public static String getLossFunctionName(ILossFunction lossFunction) {
        if (lossFunction instanceof LossNegativeLogLikelihood) {
            return "negativeloglikelihood";
        } else if (lossFunction instanceof LossMSE) {
            return "mse";
        } else if (lossFunction instanceof LossMCXENT) {
            return "mcxent";
        } else if (lossFunction instanceof LossBinaryXENT) {
            return "binary_xent";
        } else if (lossFunction instanceof LossMAE || lossFunction instanceof LossL1) {
            return "mae";
        } else if (lossFunction instanceof LossMAPE) {
            return "mape";
        } else if (lossFunction instanceof LossMSLE) {
            return "msle";
        } else if (lossFunction instanceof LossHinge) {
            return "hinge";
        } else if (lossFunction instanceof LossSquaredHinge) {
            return "squared_hinge";
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
