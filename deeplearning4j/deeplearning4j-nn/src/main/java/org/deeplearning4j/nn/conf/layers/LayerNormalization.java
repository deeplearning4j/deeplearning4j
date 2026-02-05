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
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Layer Normalization layer.
 *
 * Layer Normalization normalizes inputs across the features (last dimension by default).
 * Unlike Batch Normalization, it normalizes per sample rather than across the batch,
 * making it suitable for recurrent networks and transformers.
 *
 * y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
 *
 * Reference: "Layer Normalization" by Ba, Kiros, and Hinton (2016)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class LayerNormalization extends SameDiffLayer {

    private static final String GAMMA_KEY = "gamma";
    private static final String BETA_KEY = "beta";

    private long[] normalizedShape;
    private double epsilon;
    private boolean center;
    private boolean scale;

    private LayerNormalization() {
        // No-arg constructor for serialization
    }

    protected LayerNormalization(Builder builder) {
        super(builder);
        this.normalizedShape = builder.normalizedShape;
        this.epsilon = builder.epsilon;
        this.center = builder.center;
        this.scale = builder.scale;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (normalizedShape == null || normalizedShape.length == 0 || override) {
            // Default: normalize over the last dimension
            if (inputType.getType() == InputType.Type.FF) {
                InputType.InputTypeFeedForward f = (InputType.InputTypeFeedForward) inputType;
                this.normalizedShape = new long[]{f.getSize()};
            } else if (inputType.getType() == InputType.Type.RNN) {
                InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
                this.normalizedShape = new long[]{r.getSize()};
            } else if (inputType.getType() == InputType.Type.CNN) {
                InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
                this.normalizedShape = new long[]{c.getChannels(), c.getHeight(), c.getWidth()};
            }
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        // LayerNormalization doesn't change the shape
        return inputType;
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        if (normalizedShape != null && normalizedShape.length > 0) {
            if (scale) {
                params.addWeightParam(GAMMA_KEY, normalizedShape);
            }
            if (center) {
                params.addBiasParam(BETA_KEY, normalizedShape);
            }
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            if (scale && params.containsKey(GAMMA_KEY)) {
                INDArray gamma = params.get(GAMMA_KEY);
                gamma.assign(1.0);
            }
            if (center && params.containsKey(BETA_KEY)) {
                INDArray beta = params.get(BETA_KEY);
                beta.assign(0.0);
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput,
                                  Map<String, SDVariable> paramTable, SDVariable mask) {
        SDVariable gamma = scale ? paramTable.get(GAMMA_KEY) : null;
        SDVariable beta = center ? paramTable.get(BETA_KEY) : null;

        // Determine dimensions to normalize over
        long[] inputShape = layerInput.getShape();
        int rank = inputShape != null ? inputShape.length : 0;

        // Normalize over the last N dimensions where N = normalizedShape.length
        int numNormDims = normalizedShape != null ? normalizedShape.length : 1;
        long[] normDims = new long[numNormDims];
        for (int i = 0; i < numNormDims; i++) {
            normDims[i] = rank - numNormDims + i;
        }

        // Use the built-in layerNorm op
        SDVariable output;
        if (scale && gamma != null && center && beta != null) {
            output = sameDiff.nn().layerNorm(layerInput, gamma, beta, true, normDims);
        } else if (scale && gamma != null) {
            output = sameDiff.nn().layerNorm(layerInput, gamma, true, normDims);
        } else {
            // Manual computation if no scale/center
            SDVariable mean = sameDiff.mean(layerInput, true, normDims);
            SDVariable variance = sameDiff.variance(layerInput, false, true, normDims);
            output = layerInput.sub(mean).div(sameDiff.math().sqrt(variance.add(epsilon)));
            if (center && beta != null) {
                output = output.add(beta);
            }
        }

        return output;
    }

    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<Builder> {
        private long[] normalizedShape;
        private double epsilon = 1e-5;
        private boolean center = true;
        private boolean scale = true;

        public Builder normalizedShape(long... normalizedShape) {
            this.normalizedShape = normalizedShape;
            return this;
        }

        public Builder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        public Builder center(boolean center) {
            this.center = center;
            return this;
        }

        public Builder scale(boolean scale) {
            this.scale = scale;
            return this;
        }

        @Override
        public LayerNormalization build() {
            return new LayerNormalization(this);
        }
    }
}
