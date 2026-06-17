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
 * Cascade Attention — chunked attention for long-context KV cache.
 *
 * Splits the KV cache into chunks, computes attention per chunk,
 * then merges results using log-sum-exp for numerical stability.
 *
 * Inputs:
 *   0: query  [batch, heads, queryLen, headDim]
 *   1: key    [batch, heads, kvLen, headDim]
 *   2: value  [batch, heads, kvLen, headDim]
 *
 * Output:
 *   0: attention output [batch, heads, queryLen, headDim]
 *
 * Int args:
 *   0: chunkSize (default: 512)
 *
 * Float args:
 *   0: scale (default: 1/sqrt(headDim))
 */
@NoArgsConstructor
public class CascadeAttention extends DynamicCustomOp {

    @Getter private int chunkSize = 512;
    @Getter private double scale = 0.0;

    public CascadeAttention(INDArray query, INDArray key, INDArray value) {
        super(new INDArray[]{query, key, value}, null);
    }

    public CascadeAttention(INDArray query, INDArray key, INDArray value, int chunkSize) {
        super(new INDArray[]{query, key, value}, null);
        this.chunkSize = chunkSize;
        addIArgument((long) chunkSize);
    }

    public CascadeAttention(INDArray query, INDArray key, INDArray value, int chunkSize, double scale) {
        super(new INDArray[]{query, key, value}, null);
        this.chunkSize = chunkSize;
        this.scale = scale;
        addIArgument((long) chunkSize);
        addTArgument(scale);
    }

    public CascadeAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value) {
        super(null, sd, new SDVariable[]{query, key, value}, false);
    }

    public CascadeAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value, int chunkSize) {
        super(null, sd, new SDVariable[]{query, key, value}, false);
        this.chunkSize = chunkSize;
        addIArgument((long) chunkSize);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.chunkSize = iArguments.get(0).intValue();
        if (tArguments.size() > 0) this.scale = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "cascade_attention";
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
