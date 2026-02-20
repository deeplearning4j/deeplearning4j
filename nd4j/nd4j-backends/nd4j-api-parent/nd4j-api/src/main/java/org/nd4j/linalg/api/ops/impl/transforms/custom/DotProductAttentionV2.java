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

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


@NoArgsConstructor
public class DotProductAttentionV2 extends DynamicCustomOp {
    private boolean useCausalMask;
    private boolean training;
    private boolean useFlashAttention = true;  // Enable flash attention by default
    private double scaleFactor = 1.0;
    private double dropout = 0.0;
    private int kvCachePosition = 0;

    private SDVariable queryMask;
    private SDVariable valueMask;
    private SDVariable attentionBias;
    private SDVariable keyCache;
    private SDVariable valueCache;

    /**
     * Create a dot product attention operation.
     *
     * @param queries Query tensor [batch, Tq, dim] or [batch, Tq, heads, dim] for flash attention
     * @param values Value tensor [batch, Tv, dim] or [batch, Tv, heads, dim]
     * @param keys Key tensor [batch, Tv, dim] or [batch, Tv, heads, dim], or null to use values
     * @param queryMask Query mask [batch, Tq] or null
     * @param valueMask Value mask [batch, Tv] or null
     * @param scaleFactor Scale factor for attention scores (0 = auto: 1/sqrt(dim))
     * @param dropoutProbability Dropout probability (0 = no dropout)
     * @param useCausalMask Whether to apply causal (lower triangular) mask
     * @param training Whether in training mode (affects dropout)
     */
    public DotProductAttentionV2(INDArray queries, INDArray values, INDArray keys, INDArray queryMask, INDArray valueMask,
                                 double scaleFactor, double dropoutProbability, boolean useCausalMask, boolean training) {
        super(wrapFilterNull(queries, values, keys, queryMask, valueMask), null);
        this.scaleFactor = scaleFactor;
        this.dropout = dropoutProbability;
        this.useCausalMask = useCausalMask;
        this.training = training;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, dropoutProbability);
        // B_ARG order: useCausalMask, training, useFlashAttention
        addBArgument(useCausalMask, training, useFlashAttention);
    }

    /**
     * Create a dot product attention operation with attention bias.
     *
     * @param queries Query tensor [batch, Tq, dim] or [batch, Tq, heads, dim] for flash attention
     * @param values Value tensor [batch, Tv, dim] or [batch, Tv, heads, dim]
     * @param keys Key tensor [batch, Tv, dim] or [batch, Tv, heads, dim], or null to use values
     * @param queryMask Query mask [batch, Tq] or null
     * @param valueMask Value mask [batch, Tv] or null
     * @param attentionBias Attention bias [batch, heads, Tq, Tk] or broadcastable, added to scores before softmax
     * @param scaleFactor Scale factor for attention scores (0 = auto: 1/sqrt(dim))
     * @param dropoutProbability Dropout probability (0 = no dropout)
     * @param useCausalMask Whether to apply causal (lower triangular) mask
     * @param training Whether in training mode (affects dropout)
     */
    public DotProductAttentionV2(INDArray queries, INDArray values, INDArray keys, INDArray queryMask, INDArray valueMask,
                                 INDArray attentionBias, double scaleFactor, double dropoutProbability, 
                                 boolean useCausalMask, boolean training) {
        super(wrapFilterNull(queries, values, keys, queryMask, valueMask, attentionBias), null);
        this.scaleFactor = scaleFactor;
        this.dropout = dropoutProbability;
        this.useCausalMask = useCausalMask;
        this.training = training;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, dropoutProbability);
        // B_ARG order: useCausalMask, training, useFlashAttention
        addBArgument(useCausalMask, training, useFlashAttention);
    }

    /**
     * Full constructor for codegen compatibility.
     * Supports all parameters including KV cache and flash attention settings.
     */
    public DotProductAttentionV2(INDArray queries, INDArray values, INDArray keys,
                                 INDArray queryMask, INDArray valueMask,
                                 INDArray keyCache, INDArray valueCache,
                                 double scaleFactor, double dropoutProbability,
                                 boolean useCausalMask, boolean training,
                                 boolean useFlashAttention, int kvCachePosition) {
        super(wrapFilterNull(queries, values, keys, queryMask, valueMask, keyCache, valueCache), null);
        this.scaleFactor = scaleFactor;
        this.dropout = dropoutProbability;
        this.useCausalMask = useCausalMask;
        this.training = training;
        this.useFlashAttention = useFlashAttention;
        this.kvCachePosition = kvCachePosition;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, dropoutProbability);
        // B_ARG order: useCausalMask, training, useFlashAttention
        addBArgument(useCausalMask, training, useFlashAttention);
        // I_ARG order: kvCachePosition
        if (kvCachePosition > 0 || keyCache != null || valueCache != null) {
            addIArgument(kvCachePosition);
        }
    }

    /**
     * Create a dot product attention operation with KV cache support.
     *
     * @param queries Query tensor [batch, Tq, heads, dim]
     * @param values Value tensor [batch, Tv, heads, dim]
     * @param keys Key tensor [batch, Tv, heads, dim]
     * @param queryMask Query mask or null
     * @param valueMask Value mask or null
     * @param keyCache Key cache [batch, maxSeq, heads, dim] for autoregressive decoding
     * @param valueCache Value cache [batch, maxSeq, heads, dim] for autoregressive decoding
     * @param kvCachePosition Current position in the KV cache
     * @param scaleFactor Scale factor (0 = auto)
     * @param useCausalMask Whether to apply causal mask
     */
    public DotProductAttentionV2(INDArray queries, INDArray values, INDArray keys,
                                 INDArray queryMask, INDArray valueMask,
                                 INDArray keyCache, INDArray valueCache, int kvCachePosition,
                                 double scaleFactor, boolean useCausalMask) {
        super(wrapFilterNull(queries, values, keys, queryMask, valueMask, keyCache, valueCache), null);
        this.scaleFactor = scaleFactor;
        this.dropout = 0.0;
        this.useCausalMask = useCausalMask;
        this.training = false;
        this.kvCachePosition = kvCachePosition;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, 0.0);
        // B_ARG order: useCausalMask, training, useFlashAttention
        addBArgument(useCausalMask, false, true);
        // I_ARG order: kvCachePosition
        addIArgument(kvCachePosition);
    }

    /**
     * Create a dot product attention operation for SameDiff.
     */
    public DotProductAttentionV2(SameDiff sd, SDVariable queries, SDVariable values, SDVariable keys,
                                 SDVariable queryMask, SDVariable valueMask,
                                 double scaleFactor, double dropoutProbability,
                                 boolean useCausalMask, boolean training) {
        super(null, sd, inputs(sd, queries, values, keys, queryMask, valueMask), false);
        this.scaleFactor = scaleFactor;
        this.dropout = dropoutProbability;
        this.useCausalMask = useCausalMask;
        this.training = training;
        this.queryMask = queryMask;
        this.valueMask = valueMask;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, dropoutProbability);
        // B_ARG order: useCausalMask, training, useFlashAttention
        addBArgument(useCausalMask, training, useFlashAttention);
    }

    /**
     * Create a dot product attention operation for SameDiff with attention bias.
     *
     * @param sd SameDiff instance
     * @param queries Query tensor [batch, Tq, dim] or [batch, Tq, heads, dim] for flash attention
     * @param values Value tensor [batch, Tv, dim] or [batch, Tv, heads, dim]
     * @param keys Key tensor [batch, Tv, dim] or [batch, Tv, heads, dim], or null to use values
     * @param queryMask Query mask [batch, Tq] or null
     * @param valueMask Value mask [batch, Tv] or null
     * @param attentionBias Attention bias [batch, heads, Tq, Tk] or broadcastable, added to scores before softmax
     * @param scaleFactor Scale factor for attention scores (0 = auto: 1/sqrt(dim))
     * @param dropoutProbability Dropout probability (0 = no dropout)
     * @param useCausalMask Whether to apply causal (lower triangular) mask
     * @param training Whether in training mode (affects dropout)
     */
    public DotProductAttentionV2(SameDiff sd, SDVariable queries, SDVariable values, SDVariable keys,
                                 SDVariable queryMask, SDVariable valueMask, SDVariable attentionBias,
                                 double scaleFactor, double dropoutProbability,
                                 boolean useCausalMask, boolean training) {
        super(null, sd, inputsWithBias(sd, queries, values, keys, queryMask, valueMask, attentionBias), false);
        this.scaleFactor = scaleFactor;
        this.dropout = dropoutProbability;
        this.useCausalMask = useCausalMask;
        this.training = training;
        this.queryMask = queryMask;
        this.valueMask = valueMask;
        this.attentionBias = attentionBias;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, dropoutProbability);
        // B_ARG order: useCausalMask, training, useFlashAttention
        addBArgument(useCausalMask, training, useFlashAttention);
    }

    /**
     * Create a dot product attention operation for SameDiff with KV cache.
     */
    public DotProductAttentionV2(SameDiff sd, SDVariable queries, SDVariable values, SDVariable keys,
                                 SDVariable queryMask, SDVariable valueMask,
                                 SDVariable keyCache, SDVariable valueCache, int kvCachePosition,
                                 double scaleFactor, boolean useCausalMask) {
        super(null, sd, inputsWithCache(sd, queries, values, keys, queryMask, valueMask, keyCache, valueCache), false);
        this.scaleFactor = scaleFactor;
        this.dropout = 0.0;
        this.useCausalMask = useCausalMask;
        this.training = false;
        this.queryMask = queryMask;
        this.valueMask = valueMask;
        this.keyCache = keyCache;
        this.valueCache = valueCache;
        this.kvCachePosition = kvCachePosition;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, 0.0);
        // B_ARG order: useCausalMask, training, useFlashAttention
        addBArgument(useCausalMask, false, true);
        // I_ARG order: kvCachePosition
        addIArgument(kvCachePosition);
    }

    /**
     * Full constructor for SameDiff (for codegen compatibility).
     * Supports all parameters including KV cache and flash attention settings.
     */
    public DotProductAttentionV2(SameDiff sd, SDVariable queries, SDVariable values, SDVariable keys,
                                 SDVariable queryMask, SDVariable valueMask,
                                 SDVariable keyCache, SDVariable valueCache,
                                 double scaleFactor, double dropoutProbability,
                                 boolean useCausalMask, boolean training,
                                 boolean useFlashAttention, int kvCachePosition) {
        super(null, sd, inputsWithCache(sd, queries, values, keys, queryMask, valueMask, keyCache, valueCache), false);
        this.scaleFactor = scaleFactor;
        this.dropout = dropoutProbability;
        this.useCausalMask = useCausalMask;
        this.training = training;
        this.useFlashAttention = useFlashAttention;
        this.queryMask = queryMask;
        this.valueMask = valueMask;
        this.keyCache = keyCache;
        this.valueCache = valueCache;
        this.kvCachePosition = kvCachePosition;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, dropoutProbability);
        // B_ARG order: useCausalMask, training, useFlashAttention
        addBArgument(useCausalMask, training, useFlashAttention);
        // I_ARG order: kvCachePosition
        if (kvCachePosition > 0 || keyCache != null || valueCache != null) {
            addIArgument(kvCachePosition);
        }
    }

    private static SDVariable[] inputs(SameDiff sd,
                                       SDVariable queries,
                                       SDVariable values,
                                       SDVariable keys,
                                       SDVariable queryMask,
                                       SDVariable valueMask) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(queries);
        inputs.add(values);
        inputs.add(keys == null ? values : keys);
        inputs.add(queryMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : queryMask);
        inputs.add(valueMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : valueMask);
        return inputs.toArray(new SDVariable[inputs.size()]);
    }

    private static SDVariable[] inputsWithCache(SameDiff sd,
                                                SDVariable queries,
                                                SDVariable values,
                                                SDVariable keys,
                                                SDVariable queryMask,
                                                SDVariable valueMask,
                                                SDVariable keyCache,
                                                SDVariable valueCache) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(queries);
        inputs.add(values);
        inputs.add(keys == null ? values : keys);
        inputs.add(queryMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : queryMask);
        inputs.add(valueMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : valueMask);
        if (keyCache != null) {
            inputs.add(keyCache);
        }
        if (valueCache != null) {
            inputs.add(valueCache);
        }
        return inputs.toArray(new SDVariable[inputs.size()]);
    }

    /**
     * Build input array with attention bias (placed at index 5 per C++ op spec).
     * Input order: queries(0), values(1), keys(2), queryMask(3), valueMask(4), attentionBias(5)
     */
    private static SDVariable[] inputsWithBias(SameDiff sd,
                                               SDVariable queries,
                                               SDVariable values,
                                               SDVariable keys,
                                               SDVariable queryMask,
                                               SDVariable valueMask,
                                               SDVariable attentionBias) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(queries);
        inputs.add(values);
        inputs.add(keys == null ? values : keys);
        inputs.add(queryMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : queryMask);
        inputs.add(valueMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : valueMask);
        inputs.add(attentionBias == null ? sd.constant(Nd4j.empty(queries.dataType())) : attentionBias);
        return inputs.toArray(new SDVariable[inputs.size()]);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        // B_ARG order: useCausalMask, training, useFlashAttention
        if(bArguments.size() > 0)
            this.useCausalMask = bArguments.get(0);

        if(bArguments.size() > 1)
            this.training = bArguments.get(1);

        if(bArguments.size() > 2)
            this.useFlashAttention = bArguments.get(2);

        // T_ARG order: scale, dropout
        if(tArguments.size() > 0)
            this.scaleFactor = tArguments.get(0);

        if(tArguments.size() > 1)
            this.dropout = tArguments.get(1);

        // I_ARG order: kvCachePosition
        if(iArguments.size() > 0)
            this.kvCachePosition = iArguments.get(0).intValue();
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        super.configureWithSameDiff(sameDiff);
        if(args().length > 3)
            this.queryMask = arg(3);

        if(args().length > 4)
            this.valueMask = arg(4);

        // Input 5 can be either attentionBias or keyCache depending on context.
        // The C++ code distinguishes by checking if input 6 is present:
        // - If only input 5 is present and looks like bias shape, it's attentionBias
        // - If both input 5 and 6 are present, they are keyCache and valueCache
        if(args().length > 5) {
            if(args().length > 6) {
                // Both 5 and 6 present -> keyCache and valueCache
                this.keyCache = arg(5);
                this.valueCache = arg(6);
            } else {
                // Only 5 present -> attentionBias
                this.attentionBias = arg(5);
            }
        }
    }

    @Override
    public String opName() {
        return "dot_product_attention_v2";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        // Forward outputs: [attentionOutput, attentionWeights, attentionLogits, dropoutMask?]
        SDVariable[] outs = outputVariables();
        // BP inputs order: queries, values, keys, eps, queryMask, valueMask,
        //                  attentionScoresOut, attentionScoreWeights, attentionScoresLogits, dropoutMask,
        //                  scaleFactor, dropout, useCausalMask, training
        return Arrays.asList(new DotProductAttentionV2Bp(sameDiff,
                arg(0),                              // queries
                arg(1),                              // values
                arg(2),                              // keys
                gradient.get(0),                     // eps (gradient from upstream)
                queryMask,                           // queryMask
                valueMask,                           // valueMask
                outs[0],                             // attentionScoresOut (attention output)
                outs[1],                             // attentionScoreWeights (after softmax)
                outs[2],                             // attentionScoresLogits (before softmax)
                dropout > 0.0 ? outs[3] : null,      // dropoutMask
                scaleFactor,
                dropout,
                useCausalMask,
                training).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() >= 3,
                "Expected at least 3 input datatypes (queries, values, keys), got %s", dataTypes);

        DataType first = dataTypes.get(0);
        DataType second = dataTypes.get(1);
        DataType third = dataTypes.get(2);

        Preconditions.checkState(first != null && first.isFPType(),
                "Query datatype must be floating point, got %s", first);
        Preconditions.checkState(second != null && second.isFPType(),
                "Value datatype must be floating point, got %s", second);
        Preconditions.checkState(third != null && third.isFPType(),
                "Key datatype must be floating point, got %s", third);
        Preconditions.checkState(first == second && first == third,
                "Queries/values/keys datatypes must match, got %s", dataTypes.subList(0, 3));

        // Optional mask/bias/cache inputs may be BOOL/INT in imported graphs.
        // Runtime op handles casting/validation for optional inputs.
        if (dropout > 0)
            return Arrays.asList(first, first, first, first);

        return Arrays.asList(first, first, first);


    }

    @Override
    public int getNumOutputs() {
        return dropout > 0 ? 4 : 3;

    }
}
