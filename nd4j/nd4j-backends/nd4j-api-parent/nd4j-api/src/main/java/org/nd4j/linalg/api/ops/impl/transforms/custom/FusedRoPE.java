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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Fused Rotary Position Embedding (RoPE)
 *
 * Applies rotary position embeddings with optimized sin/cos computation.
 * Supports multiple RoPE variants used in modern LLMs:
 * - Standard (LLaMA, Mistral) - ropeType=0
 * - NeoX (GPT-NeoX, Pythia) - ropeType=1
 * - GPT-J - ropeType=2
 *
 * Input shape: [batch, seq_len, num_heads, head_dim]
 * Output shape: [batch, seq_len, num_heads, head_dim]
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FusedRoPE extends DynamicCustomOp {

    public static final int ROPE_TYPE_STANDARD = 0;  // LLaMA, Mistral
    public static final int ROPE_TYPE_NEOX = 1;       // GPT-NeoX, Pythia
    public static final int ROPE_TYPE_GPTJ = 2;       // GPT-J

    @Getter private int ropeType = ROPE_TYPE_STANDARD;
    @Getter private int positionOffset = 0;
    @Getter private double freqBase = 10000.0;
    @Getter private double freqScale = 1.0;
    @Getter private int rotaryDims = 0;  // 0 = rotate all head dims

    /**
     * Create a fused RoPE operation.
     *
     * @param sameDiff The SameDiff instance
     * @param input Input tensor [batch, seq_len, num_heads, head_dim]
     * @param ropeType RoPE variant (0=standard, 1=neox, 2=gptj)
     * @param positionOffset Starting position for KV cache continuation
     * @param freqBase Base frequency (default 10000.0)
     * @param freqScale Frequency scale factor (default 1.0)
     */
    public FusedRoPE(SameDiff sameDiff, SDVariable input, int ropeType, int positionOffset,
                     double freqBase, double freqScale) {
        this(sameDiff, input, ropeType, positionOffset, freqBase, freqScale, 0);
    }

    /**
     * Create a fused RoPE operation with partial rotation support.
     *
     * @param sameDiff The SameDiff instance
     * @param input Input tensor [batch, seq_len, num_heads, head_dim]
     * @param ropeType RoPE variant (0=standard, 1=neox, 2=gptj)
     * @param positionOffset Starting position for KV cache continuation
     * @param freqBase Base frequency (default 10000.0)
     * @param freqScale Frequency scale factor (default 1.0)
     * @param rotaryDims Number of dimensions to rotate (0 = all head dims)
     */
    public FusedRoPE(SameDiff sameDiff, SDVariable input, int ropeType, int positionOffset,
                     double freqBase, double freqScale, int rotaryDims) {
        super(null, sameDiff, new SDVariable[]{input}, false);
        this.ropeType = ropeType;
        this.positionOffset = positionOffset;
        this.freqBase = freqBase;
        this.freqScale = freqScale;
        this.rotaryDims = rotaryDims;

        addIArgument(ropeType, positionOffset, rotaryDims);
        addTArgument(freqBase, freqScale);
    }

    public FusedRoPE(SameDiff sameDiff, SDVariable input) {
        this(sameDiff, input, ROPE_TYPE_STANDARD, 0, 10000.0, 1.0);
    }

    public FusedRoPE(INDArray input, INDArray output, int ropeType, int positionOffset,
                     double freqBase, double freqScale) {
        this(input, output, ropeType, positionOffset, freqBase, freqScale, 0);
    }

    public FusedRoPE(INDArray input, INDArray output, int ropeType, int positionOffset,
                     double freqBase, double freqScale, int rotaryDims) {
        super(new INDArray[]{input}, output != null ? new INDArray[]{output} : null);
        this.ropeType = ropeType;
        this.positionOffset = positionOffset;
        this.freqBase = freqBase;
        this.freqScale = freqScale;
        this.rotaryDims = rotaryDims;

        addIArgument(ropeType, positionOffset, rotaryDims);
        addTArgument(freqBase, freqScale);
    }

    public FusedRoPE(INDArray input) {
        this(input, null, ROPE_TYPE_STANDARD, 0, 10000.0, 1.0);
    }

    /**
     * Create a fused RoPE operation with pre-computed cos/sin caches.
     * This eliminates the need for split/concat decomposition in the graph.
     *
     * @param sd The SameDiff instance
     * @param input Input tensor [batch, seq_len, num_heads, head_dim]
     * @param cosValues Pre-computed cosine values [batch, seq_len, half_head_dim]
     * @param sinValues Pre-computed sine values [batch, seq_len, half_head_dim]
     * @param ropeType RoPE variant (0=standard, 1=neox, 2=gptj)
     */
    public FusedRoPE(SameDiff sd, SDVariable input, SDVariable cosValues, SDVariable sinValues, int ropeType) {
        super(null, sd, new SDVariable[]{input, cosValues, sinValues}, false);
        this.ropeType = ropeType;
        this.positionOffset = 0;
        this.freqBase = 10000.0;
        this.freqScale = 1.0;
        addIArgument(ropeType);
    }

    /**
     * Create a fused RoPE operation with pre-computed cos/sin caches (INDArray variant).
     */
    public FusedRoPE(INDArray input, INDArray cosValues, INDArray sinValues, INDArray output, int ropeType) {
        super(new INDArray[]{input, cosValues, sinValues}, output != null ? new INDArray[]{output} : null);
        this.ropeType = ropeType;
        this.positionOffset = 0;
        this.freqBase = 10000.0;
        this.freqScale = 1.0;
        addIArgument(ropeType);
    }

    /**
     * Create a fused RoPE operation with dynamic position offset from a scalar SDVariable.
     * The position offset comes from input[1] at runtime, enabling KV cache decode
     * where position changes each step but the graph structure stays fixed.
     *
     * @param sd The SameDiff instance
     * @param input Input tensor [batch, seq_len, num_heads, head_dim]
     * @param positionOffset Scalar INT64 SDVariable with the starting position
     * @param ropeType RoPE variant (0=standard, 1=neox, 2=gptj)
     * @param freqBase Base frequency (default 10000.0)
     * @param freqScale Frequency scale factor (default 1.0)
     * @param rotaryDims Number of dimensions to rotate (0 = all head dims)
     */
    public FusedRoPE(SameDiff sd, SDVariable input, SDVariable positionOffset,
                     int ropeType, double freqBase, double freqScale, int rotaryDims) {
        super(null, sd, new SDVariable[]{input, positionOffset}, false);
        this.ropeType = ropeType;
        this.positionOffset = 0;  // dynamic — read from input[1] at runtime
        this.freqBase = freqBase;
        this.freqScale = freqScale;
        this.rotaryDims = rotaryDims;
        addIArgument(ropeType, 0, rotaryDims);
        addTArgument(freqBase, freqScale);
    }

    public FusedRoPE(SameDiff sd, SDVariable input, SDVariable ropeCache, int startPosition) {
        super(null, sd, new SDVariable[]{input, ropeCache}, false);
        this.ropeType = ROPE_TYPE_STANDARD;
        this.positionOffset = startPosition;
        this.freqBase = 10000.0;
        this.freqScale = 1.0;
        addIArgument(ropeType, positionOffset);
        addTArgument(freqBase, freqScale);
    }

    /**
     * Master SameDiff constructor used by codegen-generated SDNN methods.
     * Parameter order matches Kotlin DSL allParameters() declaration order:
     * Inputs: input, ropeCache, positionOffset
     * Args: startPosition, ropeType, freqBase, freqScale, rotaryDims
     *
     * Dispatches to the appropriate path based on which optional inputs are non-null:
     * - ropeCache != null: precomputed cache path (2-input: input + ropeCache)
     * - positionOffset != null: dynamic position path (2-input: input + positionOffset)
     * - both null: single-input path with iArg position
     */
    public FusedRoPE(SameDiff sd, SDVariable input, SDVariable ropeCache, SDVariable positionOffset,
                     int startPosition, int ropeType, double freqBase, double freqScale, int rotaryDims) {
        super(null, sd, buildRoPEInputs(input, ropeCache, positionOffset), false);
        this.ropeType = ropeType;
        this.freqBase = freqBase;
        this.freqScale = freqScale;
        this.rotaryDims = rotaryDims;

        if (positionOffset != null) {
            // Dynamic position from input tensor — position read at runtime
            this.positionOffset = 0;
            addIArgument(ropeType, 0, rotaryDims);
        } else {
            // Static position from iArg
            this.positionOffset = startPosition;
            addIArgument(ropeType, startPosition, rotaryDims);
        }
        addTArgument(freqBase, freqScale);
    }

    private static SDVariable[] buildRoPEInputs(SDVariable input, SDVariable ropeCache, SDVariable positionOffset) {
        if (ropeCache != null) {
            return new SDVariable[]{input, ropeCache};
        } else if (positionOffset != null) {
            return new SDVariable[]{input, positionOffset};
        } else {
            return new SDVariable[]{input};
        }
    }

    public FusedRoPE(INDArray input, INDArray ropeCache, int startPosition) {
        super(new INDArray[]{input, ropeCache}, null);
        this.ropeType = ROPE_TYPE_STANDARD;
        this.positionOffset = startPosition;
        this.freqBase = 10000.0;
        this.freqScale = 1.0;
        addIArgument(ropeType, positionOffset);
        addTArgument(freqBase, freqScale);
    }

    /**
     * Master INDArray constructor used by codegen-generated NDNN methods.
     * Parameter order matches Kotlin DSL allParameters() declaration order.
     */
    public FusedRoPE(INDArray input, INDArray ropeCache, INDArray positionOffset,
                     int startPosition, int ropeType, double freqBase, double freqScale, int rotaryDims) {
        super(buildINDArrayInputs(input, ropeCache, positionOffset), null);
        this.ropeType = ropeType;
        this.freqBase = freqBase;
        this.freqScale = freqScale;
        this.rotaryDims = rotaryDims;

        if (positionOffset != null) {
            this.positionOffset = 0;
            addIArgument(ropeType, 0, rotaryDims);
        } else {
            this.positionOffset = startPosition;
            addIArgument(ropeType, startPosition, rotaryDims);
        }
        addTArgument(freqBase, freqScale);
    }

    private static INDArray[] buildINDArrayInputs(INDArray input, INDArray ropeCache, INDArray positionOffset) {
        if (ropeCache != null) {
            return new INDArray[]{input, ropeCache};
        } else if (positionOffset != null) {
            return new INDArray[]{input, positionOffset};
        } else {
            return new INDArray[]{input};
        }
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.ropeType = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.positionOffset = iArguments.get(1).intValue();
        }
        if (iArguments.size() > 2) {
            this.rotaryDims = iArguments.get(2).intValue();
        }
        if (tArguments.size() > 0) {
            this.freqBase = tArguments.get(0);
        }
        if (tArguments.size() > 1) {
            this.freqScale = tArguments.get(1);
        }
    }

    @Override
    public String opName() {
        return "fused_rope";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        return Collections.singletonList(
            new FusedRoPEBp(sameDiff, arg(0), gradients.get(0), ropeType, positionOffset, freqBase, freqScale, rotaryDims).outputVariable()
        );
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
