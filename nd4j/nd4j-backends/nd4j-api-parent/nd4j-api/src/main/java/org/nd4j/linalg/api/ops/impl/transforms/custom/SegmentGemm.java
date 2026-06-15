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
 * Segmented GEMM — variable-length batched matrix multiply for MoE dispatch.
 *
 * Performs per-expert matrix multiply where each expert processes a
 * variable number of tokens.
 *
 * Inputs:
 *   0: input          [totalTokens, inDim] — concatenated token embeddings
 *   1: weights        [numExperts, inDim, outDim] — per-expert weight matrices
 *   2: segmentOffsets [numExperts] INT64 — start index per expert
 *   3: segmentSizes   [numExperts] INT64 — token count per expert
 *
 * Output:
 *   0: output [totalTokens, outDim]
 */
@NoArgsConstructor
public class SegmentGemm extends DynamicCustomOp {

    public SegmentGemm(INDArray input, INDArray weights, INDArray segmentOffsets, INDArray segmentSizes) {
        super(new INDArray[]{input, weights, segmentOffsets, segmentSizes}, null);
    }

    public SegmentGemm(SameDiff sd, SDVariable input, SDVariable weights,
                        SDVariable segmentOffsets, SDVariable segmentSizes) {
        super(null, sd, new SDVariable[]{input, weights, segmentOffsets, segmentSizes}, false);
    }

    @Override
    public String opName() {
        return "segment_gemm";
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
