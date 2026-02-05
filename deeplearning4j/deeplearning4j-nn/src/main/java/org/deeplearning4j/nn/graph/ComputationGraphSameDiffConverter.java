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
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
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

        // Mark outputs
        List<String> networkOutputs = config.getNetworkOutputs();
        for (String outputName : networkOutputs) {
            SDVariable outputVar = vertexOutputs.get(outputName);
            if (outputVar != null) {
                sd.setOutputs(outputVar.name());
            }
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
     */
    private static SDVariable convertLayer(SameDiff sd, String layerName, Layer layer,
            SDVariable input, ComputationGraph cg, DataType dataType) {

        if (layer instanceof DenseLayer) {
            return convertDenseLayer(sd, layerName, (DenseLayer) layer, input, cg, dataType);
        } else if (layer instanceof OutputLayer) {
            return convertOutputLayer(sd, layerName, (OutputLayer) layer, input, cg, dataType);
        } else if (layer instanceof ConvolutionLayer) {
            return convertConvolutionLayer(sd, layerName, (ConvolutionLayer) layer, input, cg, dataType);
        } else if (layer instanceof SubsamplingLayer) {
            return convertSubsamplingLayer(sd, layerName, (SubsamplingLayer) layer, input, dataType);
        } else if (layer instanceof BatchNormalization) {
            return convertBatchNormLayer(sd, layerName, (BatchNormalization) layer, input, cg, dataType);
        } else if (layer instanceof ActivationLayer) {
            return convertActivationLayer(sd, layerName, (ActivationLayer) layer, input);
        } else if (layer instanceof DropoutLayer) {
            // Dropout is typically handled differently in SameDiff (via config)
            return sd.identity(layerName, input);
        } else if (layer instanceof GlobalPoolingLayer) {
            return convertGlobalPoolingLayer(sd, layerName, (GlobalPoolingLayer) layer, input);
        } else if (layer instanceof EmbeddingLayer) {
            return convertEmbeddingLayer(sd, layerName, (EmbeddingLayer) layer, input, cg, dataType);
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

        // Get weights from ComputationGraph
        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = layer.hasBias() ? cgLayer.getParam(DefaultParamInitializer.BIAS_KEY) : null;

        // Create SameDiff variables for parameters
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
            OutputLayer layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = cgLayer.getParam(DefaultParamInitializer.BIAS_KEY);

        SDVariable w = sd.var(layerName + "_W", weights);
        SDVariable b = sd.var(layerName + "_b", bias);

        SDVariable z = sd.mmul(input, w).add(b);

        // Apply activation
        SDVariable output = applyActivation(sd, layerName, z, layer.getActivationFn());

        return output;
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
     * Convert a BatchNormalization layer to SameDiff operations.
     */
    private static SDVariable convertBatchNormLayer(SameDiff sd, String layerName,
            BatchNormalization layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);

        // Get parameters - gamma/beta may be null when isLockGammaBeta() is true
        INDArray gamma = cgLayer.getParam("gamma");
        INDArray beta = cgLayer.getParam("beta");
        INDArray mean = cgLayer.getParam("mean");
        // Handle isUseLogStd case - variance may be stored as log10stdev
        INDArray var = cgLayer.getParam("var");
        if (var == null) {
            var = cgLayer.getParam("log10stdev");
            if (var != null) {
                // Convert log10(stdev) to variance: var = 10^(2*log10stdev) = (10^log10stdev)^2
                var = Nd4j.math().pow(Nd4j.math().pow(Nd4j.scalar(10.0), var), 2);
            }
        }

        // Determine nOut from existing parameters or layer config
        // Try to get from an existing param first (more reliable than layer.getNOut())
        long nOut = 0;
        if (gamma != null) {
            nOut = gamma.length();
        } else if (beta != null) {
            nOut = beta.length();
        } else if (mean != null) {
            nOut = mean.length();
        } else if (var != null) {
            nOut = var.length();
        } else {
            // Fall back to layer config
            nOut = layer.getNOut();
        }

        // Ensure nOut is valid
        if (nOut <= 0) {
            throw new IllegalStateException("Could not determine nOut for BatchNormalization layer: " + layerName +
                    ". All parameters are null and layer.getNOut() returned " + nOut);
        }

        // If gamma is null (locked), use default value (layer's gamma setting, typically 1)
        if (gamma == null) {
            gamma = Nd4j.valueArrayOf(new long[]{nOut}, layer.getGamma(), dataType);
        }

        // If beta is null (locked), use default value (layer's beta setting, typically 0)
        if (beta == null) {
            beta = Nd4j.valueArrayOf(new long[]{nOut}, layer.getBeta(), dataType);
        }

        // Mean and var should always exist, but check just in case
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
            EmbeddingLayer layer, SDVariable input, ComputationGraph cg, DataType dataType) {

        org.deeplearning4j.nn.api.Layer cgLayer = cg.getLayer(layerName);
        INDArray weights = cgLayer.getParam("W");

        SDVariable embeddingMatrix = sd.var(layerName + "_W", weights);

        // Embedding lookup
        return sd.gather(layerName, embeddingMatrix, input, 0);
    }

    /**
     * Apply an activation function to a variable.
     */
    private static SDVariable applyActivation(SameDiff sd, String baseName,
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
        } else {
            log.warn("Activation {} not directly supported, using identity",
                activation.getClass().getSimpleName());
            return sd.identity(outputName, input);
        }
    }

    /**
     * Convert a SameDiff graph to a ComputationGraph.
     * Note: This is a limited conversion that works for graphs that have a structure
     * compatible with ComputationGraph (feedforward or simple recurrent architectures).
     *
     * @param sameDiff The SameDiff graph to convert
     * @param inputTypes The input types for the network
     * @return A ComputationGraph with equivalent structure
     */
    public static ComputationGraph toComputationGraph(@NonNull SameDiff sameDiff,
            InputType... inputTypes) {

        Preconditions.checkArgument(inputTypes != null && inputTypes.length > 0,
            "At least one input type must be specified");

        // This is a more complex conversion that requires pattern recognition
        // to identify layer structures in the SameDiff graph

        throw new UnsupportedOperationException(
            "SameDiff to ComputationGraph conversion is not yet fully implemented. " +
            "Use ComputationGraph.toSameDiff() for the supported direction.");
    }

    /**
     * Get the loss function corresponding to an ILossFunction.
     * Note: More specific loss functions must be checked before their parent classes.
     */
    public static String getLossFunctionName(ILossFunction lossFunction) {
        // Check specific loss functions first before their parent classes
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
