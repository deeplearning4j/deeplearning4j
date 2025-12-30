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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Unit Normalization layer (L2 normalization).
 *
 * This layer normalizes the input tensor along the specified axis so that
 * each vector along that axis has unit L2 norm.
 *
 * For input x, output = x / ||x||_2 where ||x||_2 is the L2 norm.
 *
 * This layer has no trainable parameters.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class UnitNormalization extends SameDiffLayer {

    private int axis;
    private double epsilon;

    private UnitNormalization() {
        // No-arg constructor for serialization
    }

    protected UnitNormalization(Builder builder) {
        super(builder);
        this.axis = builder.axis;
        this.epsilon = builder.epsilon;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        // No nIn to set - this layer doesn't have parameters
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        // UnitNormalization doesn't change the shape
        return inputType;
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        // No trainable parameters
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        // No parameters to initialize
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput,
                                  Map<String, SDVariable> paramTable, SDVariable mask) {
        // L2 normalization: x / sqrt(sum(x^2) + epsilon)
        // Normalize along the specified axis

        // Square the input
        SDVariable squared = sameDiff.math().square(layerInput);

        // Sum along the axis, keeping dimensions
        SDVariable sumSquared = sameDiff.sum(squared, true, axis);

        // Take square root and add epsilon for numerical stability
        SDVariable norm = sameDiff.math().sqrt(sumSquared.add(epsilon));

        // Normalize
        SDVariable output = layerInput.div(norm);

        return output;
    }

    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<Builder> {
        private int axis = -1;  // Default: last axis
        private double epsilon = 1e-12;

        public Builder axis(int axis) {
            this.axis = axis;
            return this;
        }

        public Builder epsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        @Override
        public UnitNormalization build() {
            return new UnitNormalization(this);
        }
    }
}
