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
 * Group Normalization layer.
 *
 * Group Normalization divides channels into groups and normalizes the features
 * within each group. It's an alternative to Batch Normalization that doesn't
 * depend on batch size, making it useful for small batch sizes.
 *
 * For input of shape [batch, channels, ...]:
 * 1. Divide channels into 'groups' groups
 * 2. Compute mean and variance within each group
 * 3. Normalize: (x - mean) / sqrt(variance + epsilon)
 * 4. Scale and shift: gamma * normalized + beta
 *
 * Reference: "Group Normalization" by Wu and He (2018)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class GroupNormalization extends SameDiffLayer {

    private static final String GAMMA_KEY = "gamma";
    private static final String BETA_KEY = "beta";

    private long nChannels;
    private int groups;
    private double epsilon;
    private boolean center;
    private boolean scale;

    private GroupNormalization() {
        // No-arg constructor for serialization
    }

    protected GroupNormalization(Builder builder) {
        super(builder);
        this.nChannels = builder.nChannels;
        this.groups = builder.groups;
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
        if (nChannels <= 0 || override) {
            if (inputType.getType() == InputType.Type.CNN) {
                InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
                this.nChannels = c.getChannels();
            } else if (inputType.getType() == InputType.Type.CNN3D) {
                InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
                this.nChannels = c.getChannels();
            } else if (inputType.getType() == InputType.Type.RNN) {
                InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
                this.nChannels = r.getSize();
            } else if (inputType.getType() == InputType.Type.FF) {
                InputType.InputTypeFeedForward f = (InputType.InputTypeFeedForward) inputType;
                this.nChannels = f.getSize();
            }
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        // GroupNormalization doesn't change the shape
        return inputType;
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        if (scale) {
            params.addWeightParam(GAMMA_KEY, 1, nChannels);
        }
        if (center) {
            params.addBiasParam(BETA_KEY, 1, nChannels);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            if (scale) {
                INDArray gamma = params.get(GAMMA_KEY);
                gamma.assign(1.0);
            }
            if (center) {
                INDArray beta = params.get(BETA_KEY);
                beta.assign(0.0);
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput,
                                  Map<String, SDVariable> paramTable, SDVariable mask) {
        // Get input shape
        long[] inputShape = layerInput.getShape();
        int rank = inputShape.length;

        // For simplicity, assume input is [batch, channels, ...] (NCW, NCHW, or NCDHW)
        // Reshape to [batch, groups, channels/groups, ...]

        SDVariable gamma = scale ? paramTable.get(GAMMA_KEY) : null;
        SDVariable beta = center ? paramTable.get(BETA_KEY) : null;

        // Calculate channels per group
        long channelsPerGroup = nChannels / groups;

        // Reshape input: [N, C, H, W] -> [N, G, C/G, H, W]
        // We'll reshape so we can compute stats per group
        SDVariable reshaped;
        long[] newShape;

        if (rank == 2) {
            // [batch, channels] -> [batch, groups, channels/groups]
            newShape = new long[]{-1, groups, channelsPerGroup};
        } else if (rank == 3) {
            // [batch, channels, length] -> [batch, groups, channels/groups, length]
            newShape = new long[]{-1, groups, channelsPerGroup, inputShape[2]};
        } else if (rank == 4) {
            // [batch, channels, height, width] -> [batch, groups, channels/groups, height, width]
            newShape = new long[]{-1, groups, channelsPerGroup, inputShape[2], inputShape[3]};
        } else if (rank == 5) {
            // [batch, channels, depth, height, width] -> [batch, groups, channels/groups, depth, height, width]
            newShape = new long[]{-1, groups, channelsPerGroup, inputShape[2], inputShape[3], inputShape[4]};
        } else {
            throw new IllegalStateException("GroupNormalization supports 2D, 3D, 4D, and 5D inputs. Got rank: " + rank);
        }

        reshaped = sameDiff.reshape(layerInput, newShape);

        // Compute mean and variance over all dimensions except batch and groups
        // For [N, G, C/G, H, W], normalize over dimensions 2, 3, 4
        long[] reduceDims = new long[reshaped.getShape().length - 2];
        for (int i = 0; i < reduceDims.length; i++) {
            reduceDims[i] = i + 2;
        }

        SDVariable mean = sameDiff.mean(reshaped, true, reduceDims);
        SDVariable variance = sameDiff.variance(reshaped, false, true, reduceDims);

        // Normalize
        SDVariable normalized = reshaped.sub(mean).div(sameDiff.math().sqrt(variance.add(epsilon)));

        // Reshape back to original shape
        SDVariable output = sameDiff.reshape(normalized, inputShape);

        // Apply scale and shift
        if (scale && gamma != null) {
            // Reshape gamma to broadcast: [1, channels, 1, 1, ...]
            long[] gammaShape = new long[rank];
            gammaShape[0] = 1;
            gammaShape[1] = nChannels;
            for (int i = 2; i < rank; i++) {
                gammaShape[i] = 1;
            }
            SDVariable gammaReshaped = sameDiff.reshape(gamma, gammaShape);
            output = output.mul(gammaReshaped);
        }

        if (center && beta != null) {
            // Reshape beta to broadcast: [1, channels, 1, 1, ...]
            long[] betaShape = new long[rank];
            betaShape[0] = 1;
            betaShape[1] = nChannels;
            for (int i = 2; i < rank; i++) {
                betaShape[i] = 1;
            }
            SDVariable betaReshaped = sameDiff.reshape(beta, betaShape);
            output = output.add(betaReshaped);
        }

        return output;
    }

    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<Builder> {
        private long nChannels;
        private int groups = 32;
        private double epsilon = 1e-5;
        private boolean center = true;
        private boolean scale = true;

        public Builder nChannels(long nChannels) {
            this.nChannels = nChannels;
            return this;
        }

        public Builder groups(int groups) {
            this.groups = groups;
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
        public GroupNormalization build() {
            return new GroupNormalization(this);
        }
    }
}
