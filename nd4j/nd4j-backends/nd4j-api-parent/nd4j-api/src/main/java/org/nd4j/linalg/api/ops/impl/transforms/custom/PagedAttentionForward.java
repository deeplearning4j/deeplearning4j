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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Paged attention forward pass for block-based KV cache.
 *
 * Computes scaled dot-product attention with K/V gathered from a block pool
 * via page table indirection. Optimized for single-token decode (query seqLen = 1).
 * Supports GQA (Grouped Query Attention).
 *
 * Input 0: query          [batch, 1, numHeads, headDim]
 * Input 1: keyBlockPool   [numBlocks, blockSize, numKvHeads, headDim]
 * Input 2: valueBlockPool [numBlocks, blockSize, numKvHeads, headDim]
 * Input 3: pageTables     [batch, maxBlocksPerSeq] int32
 * Input 4: contextLens    [batch] int32
 *
 * Int args:
 *   0: blockSize   — tokens per block
 *   1: numHeads    — number of query heads (0 = infer from query)
 *   2: numKvHeads  — number of KV heads for GQA (0 = same as numHeads)
 *   3: headDim     — dimension per head (0 = infer from query)
 *
 * Float args:
 *   0: scale       — attention scale (0 = auto: 1/sqrt(headDim))
 *
 * Output 0: [batch, 1, numHeads, headDim]
 */
@NoArgsConstructor
public class PagedAttentionForward extends DynamicCustomOp {

    public PagedAttentionForward(INDArray query, INDArray keyBlockPool, INDArray valueBlockPool,
                                  INDArray pageTables, INDArray contextLens,
                                  int blockSize, int numHeads, int numKvHeads, int headDim,
                                  double scale) {
        super(null, new INDArray[]{query, keyBlockPool, valueBlockPool, pageTables, contextLens}, null);
        addIArgument(blockSize, numHeads, numKvHeads, headDim);
        addTArgument(scale);
    }

    public PagedAttentionForward(INDArray query, INDArray keyBlockPool, INDArray valueBlockPool,
                                  INDArray pageTables, INDArray contextLens, int blockSize) {
        this(query, keyBlockPool, valueBlockPool, pageTables, contextLens, blockSize, 0, 0, 0, 0.0f);
    }

    public PagedAttentionForward(SameDiff sameDiff, SDVariable query, SDVariable keyBlockPool,
                                  SDVariable valueBlockPool, SDVariable pageTables, SDVariable contextLens,
                                  int blockSize, int numHeads, int numKvHeads, int headDim, double scale) {
        super(null, sameDiff, new SDVariable[]{query, keyBlockPool, valueBlockPool, pageTables, contextLens}, false);
        addIArgument(blockSize, numHeads, numKvHeads, headDim);
        addTArgument(scale);
    }

    @Override
    public String opName() {
        return "paged_attention_forward";
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
