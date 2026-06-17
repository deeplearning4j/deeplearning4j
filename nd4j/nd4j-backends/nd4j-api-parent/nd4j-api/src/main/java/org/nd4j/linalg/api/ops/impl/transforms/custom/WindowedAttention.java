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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Windowed Attention - Local/Sliding Window Attention
 *
 * Implements windowed attention mechanisms used in efficient transformers
 * like Longformer, BigBird, Swin Transformer, and SAM.
 *
 * Supports both 1D (sequence) and 2D (spatial) windowed attention.
 *
 * @author Eclipse Deeplearning4j
 */
@NoArgsConstructor
@Getter
public class WindowedAttention extends DynamicCustomOp {

    private int windowSize;
    private int numHeads;
    private int shiftSize = 0;
    private double scale = 0.0;  // 0 means auto (1/sqrt(head_dim))
    private boolean returnWeights = false;

    /**
     * Create a windowed attention operation.
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim] or [batch, height, width, num_heads, head_dim]
     * @param key Key tensor (same shape as query)
     * @param value Value tensor (same shape as query)
     * @param windowSize Size of attention window
     * @param numHeads Number of attention heads
     */
    public WindowedAttention(@NonNull INDArray query, @NonNull INDArray key,
                            @NonNull INDArray value, int windowSize, int numHeads) {
        super(new INDArray[]{query, key, value}, null);
        this.windowSize = windowSize;
        this.numHeads = numHeads;
        addArgs();
    }

    /**
     * Create a windowed attention operation with all options.
     *
     * @param query Query tensor
     * @param key Key tensor
     * @param value Value tensor
     * @param relativePositionBias Relative position bias (optional)
     * @param attentionMask Attention mask (optional)
     * @param windowSize Size of attention window
     * @param numHeads Number of attention heads
     * @param shiftSize Shift size for shifted window attention (0 for no shift)
     * @param scale Attention scale factor (0 for auto)
     * @param returnWeights Whether to return attention weights
     */
    public WindowedAttention(@NonNull INDArray query, @NonNull INDArray key,
                            @NonNull INDArray value, INDArray relativePositionBias,
                            INDArray attentionMask, int windowSize, int numHeads,
                            int shiftSize, double scale, boolean returnWeights) {
        super(buildInputs(query, key, value, relativePositionBias, attentionMask), null);
        this.windowSize = windowSize;
        this.numHeads = numHeads;
        this.shiftSize = shiftSize;
        this.scale = scale;
        this.returnWeights = returnWeights;
        addArgs();
    }

    /**
     * Create a windowed attention for SameDiff.
     *
     * @param sd SameDiff instance
     * @param query Query tensor
     * @param key Key tensor
     * @param value Value tensor
     * @param windowSize Size of attention window
     * @param numHeads Number of attention heads
     */
    public WindowedAttention(@NonNull SameDiff sd, @NonNull SDVariable query,
                            @NonNull SDVariable key, @NonNull SDVariable value,
                            int windowSize, int numHeads) {
        super(null, sd, new SDVariable[]{query, key, value}, false);
        this.windowSize = windowSize;
        this.numHeads = numHeads;
        addArgs();
    }

    /**
     * Create a windowed attention for SameDiff with all options.
     *
     * @param sd SameDiff instance
     * @param query Query tensor
     * @param key Key tensor
     * @param value Value tensor
     * @param relativePositionBias Relative position bias (optional)
     * @param attentionMask Attention mask (optional)
     * @param windowSize Size of attention window
     * @param numHeads Number of attention heads
     * @param shiftSize Shift size for shifted window attention
     * @param scale Attention scale factor
     * @param returnWeights Whether to return attention weights
     */
    public WindowedAttention(@NonNull SameDiff sd, @NonNull SDVariable query,
                            @NonNull SDVariable key, @NonNull SDVariable value,
                            SDVariable relativePositionBias, SDVariable attentionMask,
                            int windowSize, int numHeads, int shiftSize,
                            double scale, boolean returnWeights) {
        super(null, sd, buildSdInputs(query, key, value, relativePositionBias, attentionMask), false);
        this.windowSize = windowSize;
        this.numHeads = numHeads;
        this.shiftSize = shiftSize;
        this.scale = scale;
        this.returnWeights = returnWeights;
        addArgs();
    }

    private static INDArray[] buildInputs(INDArray query, INDArray key, INDArray value,
                                          INDArray relativePositionBias, INDArray attentionMask) {
        List<INDArray> inputs = new ArrayList<>();
        inputs.add(query);
        inputs.add(key);
        inputs.add(value);
        if (relativePositionBias != null) {
            inputs.add(relativePositionBias);
        }
        if (attentionMask != null) {
            inputs.add(attentionMask);
        }
        return inputs.toArray(new INDArray[0]);
    }

    private static SDVariable[] buildSdInputs(SDVariable query, SDVariable key, SDVariable value,
                                              SDVariable relativePositionBias, SDVariable attentionMask) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(query);
        inputs.add(key);
        inputs.add(value);
        if (relativePositionBias != null) {
            inputs.add(relativePositionBias);
        }
        if (attentionMask != null) {
            inputs.add(attentionMask);
        }
        return inputs.toArray(new SDVariable[0]);
    }

    private void addArgs() {
        addIArgument(windowSize, numHeads, shiftSize);
        addTArgument(scale);
        addBArgument(returnWeights);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.windowSize = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.numHeads = iArguments.get(1).intValue();
        }
        if (iArguments.size() > 2) {
            this.shiftSize = iArguments.get(2).intValue();
        }
        if (tArguments.size() > 0) {
            this.scale = tArguments.get(0);
        }
        if (bArguments.size() > 0) {
            this.returnWeights = bArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "windowed_attention";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 3,
                "Expected at least 3 input data types for %s, got %s", getClass(), inputDataTypes);
        DataType inputType = inputDataTypes.get(0);
        if (returnWeights) {
            return Arrays.asList(inputType, inputType);
        }
        return Collections.singletonList(inputType);
    }

    @Override
    public int getNumOutputs() {
        return returnWeights ? 2 : 1;
    }

    @Override
    public String onnxName() {
        return "WindowedAttention";
    }
}
