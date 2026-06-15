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

import java.util.Arrays;
import java.util.List;

/**
 * KV Cache Update - Updates key-value cache for autoregressive generation.
 *
 * This operation is used during LLM inference to efficiently store and update
 * past key-value pairs, avoiding redundant computation during token-by-token generation.
 *
 * The cache stores keys and values from previous positions, and this op
 * inserts new keys/values at the current position.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class KVCacheUpdate extends DynamicCustomOp {
    private int startPosition = 0;

    /**
     * Create a KV cache update operation.
     *
     * @param keyCache Existing key cache [batch, max_seq_len, num_kv_heads, head_dim]
     * @param valueCache Existing value cache [batch, max_seq_len, num_kv_heads, head_dim]
     * @param newKeys New keys to insert [batch, new_seq_len, num_kv_heads, head_dim]
     * @param newValues New values to insert [batch, new_seq_len, num_kv_heads, head_dim]
     * @param startPosition Position in cache where new keys/values should be inserted
     */
    public KVCacheUpdate(INDArray keyCache, INDArray valueCache,
                         INDArray newKeys, INDArray newValues, int startPosition) {
        super(wrapFilterNull(keyCache, valueCache, newKeys, newValues), null);
        this.startPosition = startPosition;
        addIArgument(startPosition);
    }

    /**
     * Create a KV cache update operation for SameDiff.
     */
    public KVCacheUpdate(SameDiff sd, SDVariable keyCache, SDVariable valueCache,
                         SDVariable newKeys, SDVariable newValues, int startPosition) {
        super(null, sd, new SDVariable[]{keyCache, valueCache, newKeys, newValues}, false);
        this.startPosition = startPosition;
        addIArgument(startPosition);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) {
            this.startPosition = iArguments.get(0).intValue();
        }
    }

    @Override
    public String opName() {
        return "kv_cache_update";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        Preconditions.checkState(first.isFPType(),
            "Input datatype must be a floating point type, got %s", first);
        // Returns updated key cache and value cache
        return Arrays.asList(first, first);
    }

    @Override
    public int getNumOutputs() {
        return 2;  // updatedKeyCache, updatedValueCache
    }
}
