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
package org.deeplearning4j.nn.modelimport.keras.layers.core;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EinsumDense;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Imports a Keras EinsumDense layer.
 *
 * EinsumDense is a dense layer that uses Einstein summation notation to define
 * the computation. It allows flexible handling of multi-dimensional inputs and
 * outputs through einsum equations.
 *
 * Keras config fields:
 * - equation: The einsum equation (e.g., "ab,bc->ac" for standard dense)
 * - output_shape: Shape of output tensor (excluding batch)
 * - bias_axes: Axes for which bias is added
 * - kernel_initializer: Weight initializer
 * - bias_initializer: Bias initializer
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasEinsumDense extends KerasLayer {

    private String equation;
    private long[] outputShape;
    private long[] kernelShape;
    private boolean hasBias;

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasEinsumDense(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasEinsumDense(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);

        // Parse equation
        this.equation = (String) innerConfig.get("equation");
        if (this.equation == null || this.equation.isEmpty()) {
            throw new InvalidKerasConfigurationException("EinsumDense layer requires 'equation' parameter");
        }

        // Parse output_shape
        Object outputShapeObj = innerConfig.get("output_shape");
        if (outputShapeObj != null) {
            this.outputShape = parseShape(outputShapeObj);
        }

        // Parse bias_axes to determine if bias is used
        Object biasAxesObj = innerConfig.get("bias_axes");
        this.hasBias = biasAxesObj != null && !biasAxesObj.toString().isEmpty();

        // Build the EinsumDense layer
        // Note: kernelShape will be set when we get the weights
        EinsumDense.Builder builder = new EinsumDense.Builder()
                .name(this.layerName)
                .equation(this.equation)
                .hasBias(this.hasBias);

        if (this.outputShape != null) {
            builder.outputShape(this.outputShape);
        }

        // Kernel shape needs to be inferred from weights or config
        // For now, set a placeholder - it will be updated when weights are loaded
        builder.kernelShape(1); // Placeholder

        this.layer = builder.build();
    }

    private long[] parseShape(Object shapeObj) {
        if (shapeObj instanceof List) {
            List<?> shapeList = (List<?>) shapeObj;
            long[] shape = new long[shapeList.size()];
            for (int i = 0; i < shapeList.size(); i++) {
                Object val = shapeList.get(i);
                if (val instanceof Number) {
                    shape[i] = ((Number) val).longValue();
                }
            }
            return shape;
        } else if (shapeObj instanceof int[]) {
            int[] intShape = (int[]) shapeObj;
            long[] shape = new long[intShape.length];
            for (int i = 0; i < intShape.length; i++) {
                shape[i] = intShape[i];
            }
            return shape;
        }
        return null;
    }

    /**
     * Get DL4J EinsumDense layer.
     *
     * @return EinsumDense
     */
    public EinsumDense getEinsumDenseLayer() {
        return (EinsumDense) this.layer;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras EinsumDense layer accepts only one input (received " + inputType.length + ")");

        if (outputShape != null && outputShape.length > 0) {
            if (outputShape.length == 1) {
                return InputType.feedForward(outputShape[0]);
            } else if (outputShape.length == 2) {
                return InputType.recurrent(outputShape[1], outputShape[0]);
            }
        }

        // Default: return same type as input
        return inputType[0];
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters
     */
    @Override
    public int getNumParams() {
        // Kernel params + optional bias params
        int numParams = 1; // At minimum, we have kernel weights
        if (hasBias) {
            numParams++;
        }
        return numParams;
    }

    /**
     * Set weights for layer.
     *
     * @param weights Map of weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();

        // Keras weight names
        String kerasKernelName = "kernel";
        String kerasBiasName = "bias";

        // Look for kernel weights
        INDArray kernel = null;
        for (Map.Entry<String, INDArray> entry : weights.entrySet()) {
            String key = entry.getKey();
            if (key.contains(kerasKernelName) || key.endsWith("/kernel:0")) {
                kernel = entry.getValue();
                break;
            }
        }

        if (kernel == null) {
            // Try alternative naming
            for (Map.Entry<String, INDArray> entry : weights.entrySet()) {
                if (!entry.getKey().contains("bias")) {
                    kernel = entry.getValue();
                    break;
                }
            }
        }

        if (kernel != null) {
            this.weights.put("W", kernel);
            this.kernelShape = kernel.shape();

            // Update the layer with correct kernel shape
            EinsumDense.Builder builder = new EinsumDense.Builder()
                    .name(this.layerName)
                    .equation(this.equation)
                    .kernelShape(this.kernelShape)
                    .hasBias(this.hasBias);

            if (this.outputShape != null) {
                builder.outputShape(this.outputShape);
            } else {
                // Infer output shape from kernel
                builder.outputShape(kernel.shape()[kernel.shape().length - 1]);
            }

            this.layer = builder.build();
        }

        // Look for bias weights
        if (hasBias) {
            INDArray bias = null;
            for (Map.Entry<String, INDArray> entry : weights.entrySet()) {
                String key = entry.getKey();
                if (key.contains(kerasBiasName) || key.endsWith("/bias:0")) {
                    bias = entry.getValue();
                    break;
                }
            }

            if (bias != null) {
                this.weights.put("b", bias);
            }
        }
    }
}
