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
 * Append new KV entries to a paged block pool via page table indirection.
 *
 * Copies new key/value tokens into the correct physical blocks as specified
 * by the page table. Handles block boundaries (tokens that span two blocks).
 *
 * Input 0: keyBlockPool   [numBlocks, blockSize, numKvHeads, headDim] (modified in-place)
 * Input 1: valueBlockPool [numBlocks, blockSize, numKvHeads, headDim] (modified in-place)
 * Input 2: newKeys        [batch, newLen, numKvHeads, headDim]
 * Input 3: newValues      [batch, newLen, numKvHeads, headDim]
 * Input 4: pageTables     [batch, maxBlocksPerSeq] int32
 * Input 5: contextLens    [batch] int32 — current valid lengths BEFORE append
 *
 * Int args:
 *   0: blockSize — tokens per block
 *
 * Output 0: scalar status (0 = success; framework requires at least 1 output)
 */
@NoArgsConstructor
public class PagedKvAppend extends DynamicCustomOp {

    public PagedKvAppend(INDArray keyBlockPool, INDArray valueBlockPool,
                          INDArray newKeys, INDArray newValues,
                          INDArray pageTables, INDArray contextLens,
                          int blockSize) {
        super(null, new INDArray[]{keyBlockPool, valueBlockPool, newKeys, newValues, pageTables, contextLens}, null);
        addIArgument(blockSize);
    }

    public PagedKvAppend(SameDiff sameDiff, SDVariable keyBlockPool, SDVariable valueBlockPool,
                          SDVariable newKeys, SDVariable newValues,
                          SDVariable pageTables, SDVariable contextLens,
                          int blockSize) {
        super(null, sameDiff, new SDVariable[]{keyBlockPool, valueBlockPool, newKeys, newValues, pageTables, contextLens}, false);
        addIArgument(blockSize);
    }

    @Override
    public String opName() {
        return "paged_kv_append";
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
