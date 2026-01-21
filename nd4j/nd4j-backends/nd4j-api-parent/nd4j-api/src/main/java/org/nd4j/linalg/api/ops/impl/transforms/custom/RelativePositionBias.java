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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Relative Position Bias - Compute relative position bias for attention
 *
 * Supports two modes:
 * 1. Learned bias (Swin/SAM style): looks up bias values from a learned table
 * 2. ALiBi: computes linear position-based bias
 *
 * @author Eclipse Deeplearning4j
 */
@NoArgsConstructor
@Getter
public class RelativePositionBias extends DynamicCustomOp {

    private int numHeads;
    private int windowSize;
    private boolean useAlibi = false;

    /**
     * Create a relative position bias operation for learned bias mode.
     *
     * @param biasTable Learned bias table [num_relative_positions, num_heads]
     * @param numHeads Number of attention heads
     * @param windowSize Window size for 2D position encoding
     */
    public RelativePositionBias(@NonNull INDArray biasTable, int numHeads, int windowSize) {
        super(new INDArray[]{biasTable}, null);
        this.numHeads = numHeads;
        this.windowSize = windowSize;
        this.useAlibi = false;
        addArgs();
    }

    /**
     * Create a relative position bias operation for learned bias with explicit index.
     *
     * @param biasTable Learned bias table [num_relative_positions, num_heads]
     * @param relativePositionIndex Precomputed relative position index [window_size^2, window_size^2]
     * @param numHeads Number of attention heads
     * @param windowSize Window size
     */
    public RelativePositionBias(@NonNull INDArray biasTable, @NonNull INDArray relativePositionIndex,
                               int numHeads, int windowSize) {
        super(new INDArray[]{biasTable, relativePositionIndex}, null);
        this.numHeads = numHeads;
        this.windowSize = windowSize;
        this.useAlibi = false;
        addArgs();
    }

    /**
     * Create a relative position bias operation for ALiBi mode.
     *
     * @param sequenceLength Scalar or tensor to infer sequence length
     * @param numHeads Number of attention heads
     * @param useAlibi Must be true to use ALiBi mode
     */
    public RelativePositionBias(@NonNull INDArray sequenceLength, int numHeads, boolean useAlibi) {
        super(new INDArray[]{sequenceLength}, null);
        this.numHeads = numHeads;
        this.windowSize = 0;  // Not used in ALiBi mode
        this.useAlibi = useAlibi;
        addArgs();
    }

    /**
     * Full constructor for codegen compatibility (INDArray version).
     *
     * @param biasTable Learned bias table
     * @param relativePositionIndex Optional precomputed relative position index (can be null)
     * @param numHeads Number of attention heads
     * @param windowSize Window size for 2D position encoding
     * @param useAlibi Whether to use ALiBi mode
     */
    public RelativePositionBias(@NonNull INDArray biasTable, INDArray relativePositionIndex,
                               int numHeads, int windowSize, boolean useAlibi) {
        super(relativePositionIndex != null ?
                new INDArray[]{biasTable, relativePositionIndex} :
                new INDArray[]{biasTable}, null);
        this.numHeads = numHeads;
        this.windowSize = windowSize;
        this.useAlibi = useAlibi;
        addArgs();
    }

    /**
     * Create a relative position bias for SameDiff (learned bias mode).
     *
     * @param sd SameDiff instance
     * @param biasTable Learned bias table
     * @param numHeads Number of attention heads
     * @param windowSize Window size
     */
    public RelativePositionBias(@NonNull SameDiff sd, @NonNull SDVariable biasTable,
                               int numHeads, int windowSize) {
        super(null, sd, new SDVariable[]{biasTable}, false);
        this.numHeads = numHeads;
        this.windowSize = windowSize;
        this.useAlibi = false;
        addArgs();
    }

    /**
     * Create a relative position bias for SameDiff (learned bias with index).
     *
     * @param sd SameDiff instance
     * @param biasTable Learned bias table
     * @param relativePositionIndex Precomputed relative position index
     * @param numHeads Number of attention heads
     * @param windowSize Window size
     */
    public RelativePositionBias(@NonNull SameDiff sd, @NonNull SDVariable biasTable,
                               @NonNull SDVariable relativePositionIndex,
                               int numHeads, int windowSize) {
        super(null, sd, new SDVariable[]{biasTable, relativePositionIndex}, false);
        this.numHeads = numHeads;
        this.windowSize = windowSize;
        this.useAlibi = false;
        addArgs();
    }

    /**
     * Create a relative position bias for SameDiff (ALiBi mode).
     *
     * @param sd SameDiff instance
     * @param sequenceLength Scalar or tensor for sequence length
     * @param numHeads Number of attention heads
     * @param useAlibi Must be true for ALiBi mode
     */
    public RelativePositionBias(@NonNull SameDiff sd, @NonNull SDVariable sequenceLength,
                               int numHeads, boolean useAlibi) {
        super(null, sd, new SDVariable[]{sequenceLength}, false);
        this.numHeads = numHeads;
        this.windowSize = 0;
        this.useAlibi = useAlibi;
        addArgs();
    }

    /**
     * Full constructor for SameDiff (for codegen compatibility).
     *
     * @param sd SameDiff instance
     * @param biasTable Learned bias table
     * @param relativePositionIndex Optional precomputed relative position index (can be null)
     * @param numHeads Number of attention heads
     * @param windowSize Window size for 2D position encoding
     * @param useAlibi Whether to use ALiBi mode
     */
    public RelativePositionBias(@NonNull SameDiff sd, @NonNull SDVariable biasTable,
                               SDVariable relativePositionIndex,
                               int numHeads, int windowSize, boolean useAlibi) {
        super(null, sd, relativePositionIndex != null ?
                new SDVariable[]{biasTable, relativePositionIndex} :
                new SDVariable[]{biasTable}, false);
        this.numHeads = numHeads;
        this.windowSize = windowSize;
        this.useAlibi = useAlibi;
        addArgs();
    }

    private void addArgs() {
        addIArgument(numHeads, windowSize, useAlibi ? 1 : 0);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.numHeads = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.windowSize = iArguments.get(1).intValue();
        }
        if (iArguments.size() > 2) {
            this.useAlibi = iArguments.get(2) != 0;
        }
    }

    @Override
    public String opName() {
        return "relative_position_bias";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 1,
                "Expected at least 1 input data type for %s, got %s", getClass(), inputDataTypes);
        DataType inputType = inputDataTypes.get(0);
        // If input is integer (for ALiBi mode with sequence length), output should be float
        if (!inputType.isFPType()) {
            return Collections.singletonList(DataType.FLOAT);
        }
        return Collections.singletonList(inputType);
    }

    @Override
    public String onnxName() {
        return "RelativePositionBias";
    }
}
