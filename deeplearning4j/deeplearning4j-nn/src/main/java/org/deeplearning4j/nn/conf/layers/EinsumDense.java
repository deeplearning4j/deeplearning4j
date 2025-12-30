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
package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.Einsum;

import java.util.*;

/**
 * EinsumDense layer - A dense layer that uses Einstein summation notation.
 *
 * This layer implements a dense transformation using einsum, allowing flexible
 * handling of multi-dimensional inputs and outputs. The equation specifies
 * how input dimensions are contracted with weight dimensions to produce output.
 *
 * Common equations:
 * - "ab,bc->ac": Standard dense layer (input: [batch, features], output: [batch, units])
 * - "abc,cd->abd": Dense on last dimension (input: [batch, seq, features], output: [batch, seq, units])
 * - "...x,xy->...y": Dense on last dimension for arbitrary input rank
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class EinsumDense extends SameDiffLayer {

    private static final String WEIGHT_KEY = "W";
    private static final String BIAS_KEY = "b";

    private String equation;
    private long[] outputShape;
    private long[] kernelShape;
    private boolean hasBias;
    private WeightInit weightInit;

    private EinsumDense() {
        // No-arg constructor for serialization
    }

    protected EinsumDense(Builder builder) {
        super(builder);
        this.equation = builder.equation;
        this.outputShape = builder.outputShape;
        this.kernelShape = builder.kernelShape;
        this.hasBias = builder.hasBias;
        this.weightInit = builder.weightInit;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        // nIn is determined by kernel shape
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        // Parse equation to determine output shape
        // For now, use the provided output shape
        if (outputShape != null && outputShape.length > 0) {
            if (outputShape.length == 1) {
                return InputType.feedForward(outputShape[0]);
            } else if (outputShape.length == 2) {
                return InputType.recurrent(outputShape[1], outputShape[0]);
            }
        }
        // Default: assume same as input with last dimension changed
        return inputType;
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam(WEIGHT_KEY, kernelShape);
        if (hasBias) {
            params.addBiasParam(BIAS_KEY, outputShape);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        // Initialize weights based on WeightInit
        // This is handled by the parent class using the weight init
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput,
                                  Map<String, SDVariable> paramTable, SDVariable mask) {
        SDVariable weights = paramTable.get(WEIGHT_KEY);

        // Parse equation to extract input and output subscripts
        String[] parts = equation.replace(" ", "").split("->");
        String inputPart = parts[0];
        String outputSubscript = parts.length > 1 ? parts[1] : "";

        String[] inputSubscripts = inputPart.split(",");

        // Perform einsum: input tensor contracted with weight tensor
        SDVariable output = sameDiff.linalg().einsum(equation, layerInput, weights);

        // Add bias if present
        if (hasBias && paramTable.containsKey(BIAS_KEY)) {
            SDVariable bias = paramTable.get(BIAS_KEY);
            output = output.add(bias);
        }

        return output;
    }

    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<Builder> {
        private String equation;
        private long[] outputShape;
        private long[] kernelShape;
        private boolean hasBias = true;
        private WeightInit weightInit = WeightInit.XAVIER;

        /**
         * Set the einsum equation.
         * The equation should have format "input_subscripts,weight_subscripts->output_subscripts"
         *
         * @param equation The einsum equation string
         */
        public Builder equation(String equation) {
            this.equation = equation;
            return this;
        }

        /**
         * Set the output shape (excluding batch dimension).
         *
         * @param outputShape The output shape
         */
        public Builder outputShape(long... outputShape) {
            this.outputShape = outputShape;
            return this;
        }

        /**
         * Set the kernel (weight) shape.
         *
         * @param kernelShape The kernel shape
         */
        public Builder kernelShape(long... kernelShape) {
            this.kernelShape = kernelShape;
            return this;
        }

        /**
         * Whether to include bias.
         *
         * @param hasBias true to include bias, false otherwise
         */
        public Builder hasBias(boolean hasBias) {
            this.hasBias = hasBias;
            return this;
        }

        /**
         * Set the weight initialization.
         *
         * @param weightInit Weight initialization to use
         */
        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        @Override
        public EinsumDense build() {
            if (equation == null || equation.isEmpty()) {
                throw new IllegalStateException("Equation must be specified for EinsumDense layer");
            }
            if (kernelShape == null || kernelShape.length == 0) {
                throw new IllegalStateException("Kernel shape must be specified for EinsumDense layer");
            }
            if (outputShape == null || outputShape.length == 0) {
                throw new IllegalStateException("Output shape must be specified for EinsumDense layer");
            }
            return new EinsumDense(this);
        }
    }
}
