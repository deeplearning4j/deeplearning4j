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
 * Scatter vision embeddings into text embeddings at target token positions.
 *
 * <p>Replaces positions in the text embedding sequence where the token ID matches
 * {@code targetTokenId} with sequential vision embeddings. Runs entirely on device
 * (no host round-trips on CUDA).</p>
 *
 * <p>Uses a two-pass approach on CUDA:
 * <ol>
 *   <li>Build exclusive prefix sum of target token matches (maps each position to its vision index)</li>
 *   <li>Scatter — each thread copies one element from vision or text based on the prefix sum</li>
 * </ol>
 *
 * <p>Inputs:
 * <ul>
 *   <li>0: textEmbeddings [batch, seqLen, hiddenDim] — text token embeddings</li>
 *   <li>1: visionEmbeddings [batch, visionTokens, hiddenDim] — vision token embeddings</li>
 *   <li>2: tokenIds [batch, seqLen] — token ID array (INT32 or INT64)</li>
 * </ul>
 *
 * <p>Integer arguments:
 * <ul>
 *   <li>0: targetTokenId — the token ID to replace with vision embeddings</li>
 * </ul>
 *
 * <p>Output: merged embeddings [batch, seqLen, hiddenDim] with vision tokens scattered in</p>
 */
@NoArgsConstructor
public class VisionEmbeddingMerge extends DynamicCustomOp {

    private long targetTokenId;

    public VisionEmbeddingMerge(SameDiff sameDiff, SDVariable textEmbeddings,
                                SDVariable visionEmbeddings, SDVariable tokenIds,
                                long targetTokenId) {
        super(null, sameDiff, new SDVariable[]{textEmbeddings, visionEmbeddings, tokenIds});
        this.targetTokenId = targetTokenId;
        addIArgument(targetTokenId);
    }

    public VisionEmbeddingMerge(INDArray textEmbeddings, INDArray visionEmbeddings,
                                INDArray tokenIds, long targetTokenId) {
        super(new INDArray[]{textEmbeddings, visionEmbeddings, tokenIds}, null);
        this.targetTokenId = targetTokenId;
        addIArgument(targetTokenId);
    }

    @Override
    public String opName() {
        return "vision_embedding_merge";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
