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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * ONNX MultiHeadAttention operation.
 * 
 * Takes pre-projected Q, K, V tensors (already projected, unlike MultiHeadDotProductAttention
 * which requires weight matrices).
 * 
 * Compatible with Microsoft's ONNX MultiHeadAttention operator.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class OnnxMultiHeadAttention extends DynamicCustomOp {
    private int numHeads;
    private boolean useCausalMask;
    private double scale;
    private int numOutputs = 3;

    /**
     * SameDiff constructor for ONNX MultiHeadAttention.
     *
     * @param sd           SameDiff instance
     * @param query        Query tensor [batch, seqQ, hidden]
     * @param key          Key tensor [batch, seqKV, hidden]
     * @param value        Value tensor [batch, seqKV, hidden]
     * @param attnBias     Optional attention bias [batch, numHeads, seqQ, seqKV] or null
     * @param pastKey      Optional past key [batch, numHeads, pastSeq, headDim] or null
     * @param pastValue    Optional past value [batch, numHeads, pastSeq, headDim] or null
     * @param numHeads     Number of attention heads
     * @param scale        Scale factor (0 = auto compute 1/sqrt(headDim))
     * @param useCausalMask Whether to apply causal masking
     */
    public OnnxMultiHeadAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                                   SDVariable attnBias, SDVariable pastKey, SDVariable pastValue,
                                   int numHeads, double scale, boolean useCausalMask) {
        this(sd, query, key, value, attnBias, pastKey, pastValue, numHeads, scale, useCausalMask, 3);
    }

    /**
     * SameDiff constructor with explicit output count.
     * Use 1 when ONNX optional present_key/present_value outputs are empty.
     */
    public OnnxMultiHeadAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                                  SDVariable attnBias, SDVariable pastKey, SDVariable pastValue,
                                  int numHeads, double scale, boolean useCausalMask, int numOutputs) {
        super(null, sd, buildInputs(sd, query, key, value, attnBias, pastKey, pastValue), false);
        this.numHeads = numHeads;
        this.scale = scale;
        this.useCausalMask = useCausalMask;
        Preconditions.checkArgument(numOutputs >= 1 && numOutputs <= 3,
                "numOutputs must be in [1, 3], got %s", numOutputs);
        this.numOutputs = numOutputs;
        addIArgument(numHeads);
        addIArgument(useCausalMask ? 1 : 0);
        addTArgument(scale);
    }

    /**
     * INDArray constructor for ONNX MultiHeadAttention.
     */
    public OnnxMultiHeadAttention(INDArray query, INDArray key, INDArray value,
                                   INDArray attnBias, INDArray pastKey, INDArray pastValue,
                                   int numHeads, double scale, boolean useCausalMask) {
        this(query, key, value, attnBias, pastKey, pastValue, numHeads, scale, useCausalMask, 3);
    }

    /**
     * INDArray constructor with explicit output count.
     */
    public OnnxMultiHeadAttention(INDArray query, INDArray key, INDArray value,
                                  INDArray attnBias, INDArray pastKey, INDArray pastValue,
                                  int numHeads, double scale, boolean useCausalMask, int numOutputs) {
        super(buildInputs(query, key, value, attnBias, pastKey, pastValue), null);
        this.numHeads = numHeads;
        this.scale = scale;
        this.useCausalMask = useCausalMask;
        Preconditions.checkArgument(numOutputs >= 1 && numOutputs <= 3,
                "numOutputs must be in [1, 3], got %s", numOutputs);
        this.numOutputs = numOutputs;
        addIArgument(numHeads);
        addIArgument(useCausalMask ? 1 : 0);
        addTArgument(scale);
    }

    private static SDVariable[] buildInputs(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                                            SDVariable attnBias, SDVariable pastKey, SDVariable pastValue) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(query);
        inputs.add(key);
        inputs.add(value);
        inputs.add(attnBias != null ? attnBias : sd.constant(Nd4j.empty(query.dataType())));
        inputs.add(pastKey != null ? pastKey : sd.constant(Nd4j.empty(query.dataType())));
        inputs.add(pastValue != null ? pastValue : sd.constant(Nd4j.empty(query.dataType())));
        return inputs.toArray(new SDVariable[0]);
    }

    private static INDArray[] buildInputs(INDArray query, INDArray key, INDArray value,
                                          INDArray attnBias, INDArray pastKey, INDArray pastValue) {
        // Always build 6 inputs to maintain positional semantics for C++ op
        // Use scalar placeholder (not empty!) for missing optional inputs to avoid null buffer issues
        INDArray placeholder = Nd4j.scalar(query.dataType(), 0.0f);
        
        return new INDArray[] {
            query,
            key,
            value,
            attnBias != null ? attnBias : placeholder,
            pastKey != null ? pastKey : placeholder,
            pastValue != null ? pastValue : placeholder
        };
    }

    @Override
    public String opName() {
        return "onnx_multi_head_attention";
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.numHeads = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.useCausalMask = iArguments.get(1).intValue() != 0;
        }
        if (tArguments.size() > 0) {
            this.scale = tArguments.get(0);
        }
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        super.setPropertiesForFunction(properties);
        Object nOut = properties.get("numOutputs");
        if (nOut instanceof Number) {
            this.numOutputs = ((Number) nOut).intValue();
        }
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        // Descriptor fallback for local ops that may not yet exist in nd4j-op-def.pbtxt.
        // This keeps clone/serialization paths working for graph optimizer and SDZ caching.
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("numHeads", numHeads);
        ret.put("useCausalMask", useCausalMask);
        ret.put("scale", scale);
        ret.put("numOutputs", numOutputs);
        return ret;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() >= 3,
                "Expected at least 3 input datatypes, got %s", dataTypes);
        DataType first = dataTypes.get(0);
        Preconditions.checkState(first.isFPType(),
                "Query datatype must be floating point, got %s", first);
        if (numOutputs == 1) {
            return Arrays.asList(first);
        } else if (numOutputs == 2) {
            return Arrays.asList(first, first);
        }
        return Arrays.asList(first, first, first);
    }

    @Override
    public int getNumOutputs() {
        return numOutputs;
    }
}
