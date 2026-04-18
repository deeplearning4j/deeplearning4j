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

package org.nd4j.linalg.dataset.curation.batching;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Holds the result of a padding-free batching operation.
 *
 * <p>In padding-free batching, all sequences in a batch are concatenated
 * into a single long sequence without padding, and their boundaries are
 * tracked via cumulative sequence lengths ({@code cuSeqlens}).
 *
 * <p>Fields:
 * <ul>
 *   <li>{@code unpadded} — [total_tokens, hidden_dim]: concatenated real tokens</li>
 *   <li>{@code indices} — [total_tokens] INT64: flat index into the original padded
 *       batch for each real token (used to reverse the operation via
 *       {@link org.nd4j.linalg.api.ops.impl.transforms.custom.PadInput})</li>
 *   <li>{@code cuSeqlens} — [batch+1] INT64: cumulative sequence lengths starting
 *       with 0, e.g. [0, 3, 7] for a batch of two sequences with lengths 3 and 4</li>
 *   <li>{@code maxSeqlenInBatch} — maximum actual sequence length across the batch</li>
 *   <li>{@code batchSize} — number of sequences in the batch</li>
 * </ul>
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PaddingFreeBatch {

    /** [total_tokens, hidden_dim] — concatenated real tokens, no padding. */
    private INDArray unpadded;

    /**
     * [total_tokens] INT64 — flat index into the original padded batch for
     * each real token. Index = batchIdx * maxSeqLen + seqPos.
     */
    private INDArray indices;

    /**
     * [batch+1] INT64 — cumulative sequence lengths starting with 0.
     * cuSeqlens[i+1] - cuSeqlens[i] = length of sequence i.
     */
    private INDArray cuSeqlens;

    /** Maximum actual sequence length in this batch. */
    private int maxSeqlenInBatch;

    /** Number of sequences (batch dimension). */
    private int batchSize;
}
