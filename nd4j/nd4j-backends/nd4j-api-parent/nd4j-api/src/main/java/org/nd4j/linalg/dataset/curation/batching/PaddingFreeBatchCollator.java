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

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.PadInput;
import org.nd4j.linalg.api.ops.impl.transforms.custom.UnpadInput;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

/**
 * Utility class for padding-free batching.
 *
 * <p>Padding-free batching concatenates all sequences in a batch into a single
 * long sequence, eliminating wasted compute on padding tokens. Sequence
 * boundaries are preserved via cumulative sequence lengths ({@code cu_seqlens}).
 *
 * <p>Three entry points are provided:
 * <ul>
 *   <li>{@link #collate(List)} — build from a list of variable-length sequences</li>
 *   <li>{@link #fromPadded(INDArray, INDArray)} — convert an already-padded batch
 *       using its attention mask (calls the native {@code unpad_input} C++ op)</li>
 *   <li>{@link #toPadded(PaddingFreeBatch)} — restore padded format from a
 *       {@link PaddingFreeBatch} (calls the native {@code pad_input} C++ op)</li>
 * </ul>
 *
 * @author Adam Gibson
 */
public final class PaddingFreeBatchCollator {

    private PaddingFreeBatchCollator() {}

    // -------------------------------------------------------------------------
    // collate: build from a list of variable-length sequences
    // -------------------------------------------------------------------------

    /**
     * Collates a list of variable-length 2-D sequences (each shaped
     * [{@code seq_len_i}, {@code hidden_dim}]) into a {@link PaddingFreeBatch}.
     *
     * <p>The sequences are concatenated along axis 0 in the order given. The
     * cumulative sequence lengths and flat indices are computed here in Java
     * (no padded intermediate is created).
     *
     * @param sequences list of [seq_len_i, hidden_dim] arrays (all same hidden_dim)
     * @return {@link PaddingFreeBatch} with all metadata filled
     */
    public static PaddingFreeBatch collate(List<INDArray> sequences) {
        Preconditions.checkArgument(sequences != null && !sequences.isEmpty(),
                "PaddingFreeBatchCollator.collate: sequences must be non-null and non-empty");

        int batchSize   = sequences.size();
        long hiddenDim  = sequences.get(0).size(1);
        long totalTokens = 0;
        long maxSeqLen   = 0;

        for (INDArray seq : sequences) {
            Preconditions.checkArgument(seq.rank() == 2,
                    "PaddingFreeBatchCollator.collate: each sequence must be rank 2, got rank %d",
                    seq.rank());
            Preconditions.checkArgument(seq.size(1) == hiddenDim,
                    "PaddingFreeBatchCollator.collate: hidden_dim mismatch: expected %d, got %d",
                    hiddenDim, seq.size(1));
            long len = seq.size(0);
            totalTokens += len;
            if (len > maxSeqLen) maxSeqLen = len;
        }

        // Concatenate sequences
        INDArray unpadded = Nd4j.vstack(sequences.toArray(new INDArray[0]));

        // Build cu_seqlens
        INDArray cuSeqlens = Nd4j.zeros(DataType.INT64, batchSize + 1);
        long cumSum = 0;
        cuSeqlens.putScalar(0, 0L);
        for (int i = 0; i < batchSize; i++) {
            cumSum += sequences.get(i).size(0);
            cuSeqlens.putScalar(i + 1, cumSum);
        }

        // Build indices: for each real token, its flat position in a hypothetical
        // padded batch of shape [batch, maxSeqLen, hiddenDim].
        // For sequence b with actual length L_b, token t maps to b*maxSeqLen + t.
        INDArray indices = Nd4j.zeros(DataType.INT64, totalTokens);
        long outIdx = 0;
        for (int b = 0; b < batchSize; b++) {
            long seqLen = sequences.get(b).size(0);
            for (long t = 0; t < seqLen; t++) {
                indices.putScalar(outIdx++, (long) b * maxSeqLen + t);
            }
        }

        return PaddingFreeBatch.builder()
                .unpadded(unpadded)
                .indices(indices)
                .cuSeqlens(cuSeqlens)
                .maxSeqlenInBatch((int) maxSeqLen)
                .batchSize(batchSize)
                .build();
    }

    // -------------------------------------------------------------------------
    // fromPadded: unpad an existing padded batch via the native C++ op
    // -------------------------------------------------------------------------

    /**
     * Creates a {@link PaddingFreeBatch} from an already-padded batch tensor and
     * its corresponding attention mask, using the native {@code unpad_input} C++ op.
     *
     * @param paddedBatch   [batch, max_seq_len, hidden_dim] padded hidden states
     * @param attentionMask [batch, max_seq_len] attention mask (1=real, 0=pad)
     * @return {@link PaddingFreeBatch} containing the compact representation
     */
    public static PaddingFreeBatch fromPadded(INDArray paddedBatch, INDArray attentionMask) {
        Preconditions.checkArgument(paddedBatch != null, "paddedBatch must not be null");
        Preconditions.checkArgument(attentionMask != null, "attentionMask must not be null");
        Preconditions.checkArgument(paddedBatch.rank() == 3,
                "paddedBatch must be rank 3 [batch, max_seq_len, hidden_dim], got rank %d",
                paddedBatch.rank());
        Preconditions.checkArgument(attentionMask.rank() == 2,
                "attentionMask must be rank 2 [batch, max_seq_len], got rank %d",
                attentionMask.rank());

        INDArray[] outputs = Nd4j.exec(new UnpadInput(paddedBatch, attentionMask));

        // outputs: [unpadded, indices, cu_seqlens, max_seqlen_scalar]
        INDArray unpadded   = outputs[0];
        INDArray indices    = outputs[1];
        INDArray cuSeqlens  = outputs[2];
        INDArray maxSeqScal = outputs[3];

        // Determine actual total_tokens: cu_seqlens[-1]
        int batchSize = (int) paddedBatch.size(0);
        long totalTokens = cuSeqlens.getLong(batchSize);

        // Trim unpadded and indices to actual size (C++ op uses worst-case shape)
        INDArray unpaddedTrimmed = totalTokens < unpadded.size(0)
                ? unpadded.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, totalTokens),
                               org.nd4j.linalg.indexing.NDArrayIndex.all())
                : unpadded;
        INDArray indicesTrimmed = totalTokens < indices.size(0)
                ? indices.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, totalTokens))
                : indices;

        int maxSeqLen = (int) maxSeqScal.getLong(0);

        return PaddingFreeBatch.builder()
                .unpadded(unpaddedTrimmed)
                .indices(indicesTrimmed)
                .cuSeqlens(cuSeqlens)
                .maxSeqlenInBatch(maxSeqLen)
                .batchSize(batchSize)
                .build();
    }

    // -------------------------------------------------------------------------
    // toPadded: restore padded format via the native C++ op
    // -------------------------------------------------------------------------

    /**
     * Restores the padded [batch, max_seq_len, hidden_dim] format from a
     * {@link PaddingFreeBatch}, using the native {@code pad_input} C++ op.
     *
     * <p>Padding positions are filled with zeros.
     *
     * @param batch the padding-free batch to restore
     * @return padded [batch, max_seq_len, hidden_dim] tensor
     */
    public static INDArray toPadded(PaddingFreeBatch batch) {
        Preconditions.checkArgument(batch != null, "batch must not be null");
        Preconditions.checkArgument(batch.getUnpadded() != null, "batch.unpadded must not be null");
        Preconditions.checkArgument(batch.getIndices() != null, "batch.indices must not be null");

        INDArray batchSizeScalar = Nd4j.scalar(DataType.INT64, (long) batch.getBatchSize());
        INDArray maxSeqLenScalar = Nd4j.scalar(DataType.INT64, (long) batch.getMaxSeqlenInBatch());

        INDArray[] outputs = Nd4j.exec(new PadInput(
                batch.getUnpadded(),
                batch.getIndices(),
                batchSizeScalar,
                maxSeqLenScalar));

        return outputs[0];
    }
}
