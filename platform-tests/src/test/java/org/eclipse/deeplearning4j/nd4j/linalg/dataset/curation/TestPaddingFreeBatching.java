package org.eclipse.deeplearning4j.nd4j.linalg.dataset.curation;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.PadInput;
import org.nd4j.linalg.api.ops.impl.transforms.custom.PadInputBp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.UnpadInput;
import org.nd4j.linalg.dataset.curation.batching.PaddingFreeBatch;
import org.nd4j.linalg.dataset.curation.batching.PaddingFreeBatchCollator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for padding-free batching via unpad_input / pad_input C++ ops
 * and the PaddingFreeBatchCollator Java utility.
 */
public class TestPaddingFreeBatching {

    // ------------------------------------------------------------------
    // Helper: build a padded [batch, maxSeqLen, hiddenDim] tensor
    // where hidden_states[b, s, :] = (b * 100 + s + 1) for real tokens,
    // and 0 for padding positions.
    // ------------------------------------------------------------------
    private static INDArray buildPaddedHidden(int batch, int maxSeqLen, int hiddenDim, int[] realLens) {
        INDArray out = Nd4j.zeros(DataType.FLOAT, batch, maxSeqLen, hiddenDim);
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < realLens[b]; s++) {
                float val = b * 100f + s + 1f;
                for (int d = 0; d < hiddenDim; d++) {
                    out.putScalar(new int[]{b, s, d}, val + d * 0.01f);
                }
            }
        }
        return out;
    }

    private static INDArray buildMask(int batch, int maxSeqLen, int[] realLens) {
        INDArray mask = Nd4j.zeros(DataType.INT32, batch, maxSeqLen);
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < realLens[b]; s++) {
                mask.putScalar(new int[]{b, s}, 1);
            }
        }
        return mask;
    }

    // ------------------------------------------------------------------
    // testUnpadInput: batch=[2,5,8], mask=[[1,1,1,0,0],[1,1,1,1,0]]
    // Expected: unpadded shape [7,8], cu_seqlens=[0,3,7], max_seqlen=4
    // ------------------------------------------------------------------
    @Test
    public void testUnpadInput() {
        int batch = 2, maxSeqLen = 5, hiddenDim = 8;
        int[] realLens = {3, 4};

        INDArray hidden = buildPaddedHidden(batch, maxSeqLen, hiddenDim, realLens);
        INDArray mask   = buildMask(batch, maxSeqLen, realLens);

        INDArray[] outputs = Nd4j.exec(new UnpadInput(hidden, mask));
        INDArray unpadded  = outputs[0];
        INDArray indices   = outputs[1];
        INDArray cuSeqlens = outputs[2];
        INDArray maxSeqScal = outputs[3];

        // Trim to actual total_tokens = 7
        long totalTokens = cuSeqlens.getLong(batch);
        assertEquals(7L, totalTokens, "total tokens should be 7");

        // Shape checks
        assertEquals(2, unpadded.rank(), "unpadded should be rank 2");
        // first dim may be max-case padded; verify the first 7 rows are correct
        assertEquals((long) hiddenDim, unpadded.size(1), "unpadded hidden_dim mismatch");

        // cu_seqlens
        assertEquals(3, cuSeqlens.size(0), "cu_seqlens should have batch+1=3 elements");
        assertEquals(0L, cuSeqlens.getLong(0), "cu_seqlens[0] should be 0");
        assertEquals(3L, cuSeqlens.getLong(1), "cu_seqlens[1] should be 3");
        assertEquals(7L, cuSeqlens.getLong(2), "cu_seqlens[2] should be 7");

        // max_seqlen
        assertEquals(4L, maxSeqScal.getLong(0), "max_seqlen should be 4");

        // Verify first real token of seq 0 = hidden[0,0,:]
        for (int d = 0; d < hiddenDim; d++) {
            float expected = hidden.getFloat(new int[]{0, 0, d});
            float actual   = unpadded.getFloat(new long[]{0, d});
            assertEquals(expected, actual, 1e-5f,
                    "unpadded token 0 dim " + d + " mismatch");
        }

        // Verify first real token of seq 1 = hidden[1,0,:] (at row 3)
        for (int d = 0; d < hiddenDim; d++) {
            float expected = hidden.getFloat(new int[]{1, 0, d});
            float actual   = unpadded.getFloat(new long[]{3, d});
            assertEquals(expected, actual, 1e-5f,
                    "unpadded token 3 (seq1, pos0) dim " + d + " mismatch");
        }
    }

    // ------------------------------------------------------------------
    // testPadInput: reverse of the above, verify padded output
    // ------------------------------------------------------------------
    @Test
    public void testPadInput() {
        int batch = 2, maxSeqLen = 5, hiddenDim = 8;
        int[] realLens = {3, 4};

        INDArray hidden = buildPaddedHidden(batch, maxSeqLen, hiddenDim, realLens);
        INDArray mask   = buildMask(batch, maxSeqLen, realLens);

        INDArray[] unpadOutputs = Nd4j.exec(new UnpadInput(hidden, mask));
        INDArray unpadded  = unpadOutputs[0];
        INDArray indices   = unpadOutputs[1];
        INDArray cuSeqlens = unpadOutputs[2];

        long totalTokens = cuSeqlens.getLong(batch);
        INDArray unpaddedTrimmed = totalTokens < unpadded.size(0)
                ? unpadded.get(NDArrayIndex.interval(0, totalTokens), NDArrayIndex.all())
                : unpadded;
        INDArray indicesTrimmed = totalTokens < indices.size(0)
                ? indices.get(NDArrayIndex.interval(0, totalTokens))
                : indices;

        INDArray[] padOutputs = Nd4j.exec(new PadInput(
                unpaddedTrimmed,
                indicesTrimmed,
                Nd4j.scalar(DataType.INT64, (long) batch),
                Nd4j.scalar(DataType.INT64, (long) maxSeqLen)));

        INDArray padded = padOutputs[0];

        assertEquals(3, padded.rank(), "padded should be rank 3");
        assertEquals((long) batch, padded.size(0));
        assertEquals((long) maxSeqLen, padded.size(1));
        assertEquals((long) hiddenDim, padded.size(2));

        // Real token positions should match original
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < realLens[b]; s++) {
                for (int d = 0; d < hiddenDim; d++) {
                    float expected = hidden.getFloat(new int[]{b, s, d});
                    float actual   = padded.getFloat(new int[]{b, s, d});
                    assertEquals(expected, actual, 1e-5f,
                            "padded[" + b + "," + s + "," + d + "] mismatch");
                }
            }
        }

        // Padding positions should be 0
        for (int b = 0; b < batch; b++) {
            for (int s = realLens[b]; s < maxSeqLen; s++) {
                for (int d = 0; d < hiddenDim; d++) {
                    assertEquals(0f, padded.getFloat(new int[]{b, s, d}), 1e-6f,
                            "padding[" + b + "," + s + "," + d + "] should be 0");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // testRoundTrip: unpad then pad should recover the original padded tensor
    // ------------------------------------------------------------------
    @Test
    public void testRoundTrip() {
        int batch = 3, maxSeqLen = 6, hiddenDim = 4;
        int[] realLens = {2, 5, 3};

        INDArray hidden = buildPaddedHidden(batch, maxSeqLen, hiddenDim, realLens);
        INDArray mask   = buildMask(batch, maxSeqLen, realLens);

        INDArray[] unpadOutputs = Nd4j.exec(new UnpadInput(hidden, mask));
        long totalTokens = unpadOutputs[2].getLong(batch);

        INDArray unpaddedTrimmed = totalTokens < unpadOutputs[0].size(0)
                ? unpadOutputs[0].get(NDArrayIndex.interval(0, totalTokens), NDArrayIndex.all())
                : unpadOutputs[0];
        INDArray indicesTrimmed = totalTokens < unpadOutputs[1].size(0)
                ? unpadOutputs[1].get(NDArrayIndex.interval(0, totalTokens))
                : unpadOutputs[1];

        INDArray[] padOutputs = Nd4j.exec(new PadInput(
                unpaddedTrimmed, indicesTrimmed,
                Nd4j.scalar(DataType.INT64, (long) batch),
                Nd4j.scalar(DataType.INT64, (long) maxSeqLen)));

        INDArray recovered = padOutputs[0];

        // Real positions must match
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < realLens[b]; s++) {
                for (int d = 0; d < hiddenDim; d++) {
                    assertEquals(hidden.getFloat(new int[]{b, s, d}),
                                 recovered.getFloat(new int[]{b, s, d}),
                                 1e-5f,
                                 "Round-trip mismatch at [" + b + "," + s + "," + d + "]");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // testUniformLengths: all sequences same length, cu_seqlens evenly spaced
    // ------------------------------------------------------------------
    @Test
    public void testUniformLengths() {
        int batch = 4, maxSeqLen = 3, hiddenDim = 5;
        int[] realLens = {3, 3, 3, 3};

        INDArray hidden = buildPaddedHidden(batch, maxSeqLen, hiddenDim, realLens);
        INDArray mask   = buildMask(batch, maxSeqLen, realLens);

        INDArray[] outputs = Nd4j.exec(new UnpadInput(hidden, mask));
        INDArray cuSeqlens = outputs[2];

        assertEquals(5, cuSeqlens.size(0), "cu_seqlens should have batch+1=5 elements");
        for (int i = 0; i <= batch; i++) {
            assertEquals((long) i * 3, cuSeqlens.getLong(i),
                    "cu_seqlens[" + i + "] should be " + (i * 3));
        }
        assertEquals(3L, outputs[3].getLong(0), "max_seqlen should be 3");
    }

    // ------------------------------------------------------------------
    // testSingleSequence: batch size 1
    // ------------------------------------------------------------------
    @Test
    public void testSingleSequence() {
        int batch = 1, maxSeqLen = 4, hiddenDim = 6;
        int[] realLens = {3};

        INDArray hidden = buildPaddedHidden(batch, maxSeqLen, hiddenDim, realLens);
        INDArray mask   = buildMask(batch, maxSeqLen, realLens);

        INDArray[] outputs = Nd4j.exec(new UnpadInput(hidden, mask));
        INDArray cuSeqlens = outputs[2];
        long totalTokens = cuSeqlens.getLong(batch);

        assertEquals(3L, totalTokens, "total_tokens should be 3 for single seq of length 3");
        assertEquals(2, cuSeqlens.size(0), "cu_seqlens should have 2 elements for batch=1");
        assertEquals(0L, cuSeqlens.getLong(0));
        assertEquals(3L, cuSeqlens.getLong(1));
        assertEquals(3L, outputs[3].getLong(0), "max_seqlen should be 3");
    }

    // ------------------------------------------------------------------
    // testAllPadding: degenerate case — mask all zeros
    // ------------------------------------------------------------------
    @Test
    public void testAllPadding() {
        int batch = 2, maxSeqLen = 3, hiddenDim = 4;
        int[] realLens = {0, 0};

        INDArray hidden = Nd4j.zeros(DataType.FLOAT, batch, maxSeqLen, hiddenDim);
        INDArray mask   = Nd4j.zeros(DataType.INT32, batch, maxSeqLen);

        INDArray[] outputs = Nd4j.exec(new UnpadInput(hidden, mask));
        INDArray cuSeqlens = outputs[2];
        long totalTokens = cuSeqlens.getLong(batch);

        assertEquals(0L, totalTokens, "total_tokens should be 0 when all padding");
        assertEquals(0L, cuSeqlens.getLong(0));
        assertEquals(0L, cuSeqlens.getLong(1));
        assertEquals(0L, cuSeqlens.getLong(2));
        assertEquals(0L, outputs[3].getLong(0), "max_seqlen should be 0");
    }

    // ------------------------------------------------------------------
    // testPaddingFreeBatchCollator: Java collator with list of sequences
    // ------------------------------------------------------------------
    @Test
    public void testPaddingFreeBatchCollator() {
        int hiddenDim = 4;

        // Three sequences of different lengths
        INDArray seq0 = Nd4j.ones(DataType.FLOAT, 3, hiddenDim).muli(1f);
        INDArray seq1 = Nd4j.ones(DataType.FLOAT, 5, hiddenDim).muli(2f);
        INDArray seq2 = Nd4j.ones(DataType.FLOAT, 2, hiddenDim).muli(3f);

        List<INDArray> seqs = Arrays.asList(seq0, seq1, seq2);
        PaddingFreeBatch batch = PaddingFreeBatchCollator.collate(seqs);

        assertNotNull(batch);
        assertEquals(3, batch.getBatchSize());
        assertEquals(5, batch.getMaxSeqlenInBatch());
        assertEquals(10L, batch.getUnpadded().size(0), "total_tokens should be 3+5+2=10");
        assertEquals((long) hiddenDim, batch.getUnpadded().size(1));

        // cu_seqlens: [0, 3, 8, 10]
        INDArray cu = batch.getCuSeqlens();
        assertEquals(4, cu.size(0));
        assertEquals(0L, cu.getLong(0));
        assertEquals(3L, cu.getLong(1));
        assertEquals(8L, cu.getLong(2));
        assertEquals(10L, cu.getLong(3));

        // Verify content: first 3 rows should be all 1s
        for (int t = 0; t < 3; t++) {
            for (int d = 0; d < hiddenDim; d++) {
                assertEquals(1f, batch.getUnpadded().getFloat(new long[]{t, d}), 1e-5f,
                        "seq0 token " + t + " dim " + d + " should be 1");
            }
        }
        // rows 3-7 should be all 2s
        for (int t = 3; t < 8; t++) {
            for (int d = 0; d < hiddenDim; d++) {
                assertEquals(2f, batch.getUnpadded().getFloat(new long[]{t, d}), 1e-5f,
                        "seq1 token " + t + " dim " + d + " should be 2");
            }
        }
        // rows 8-9 should be all 3s
        for (int t = 8; t < 10; t++) {
            for (int d = 0; d < hiddenDim; d++) {
                assertEquals(3f, batch.getUnpadded().getFloat(new long[]{t, d}), 1e-5f,
                        "seq2 token " + t + " dim " + d + " should be 3");
            }
        }
    }

    // ------------------------------------------------------------------
    // testFromPaddedAndToPadded: convenience collator methods round-trip
    // ------------------------------------------------------------------
    @Test
    public void testFromPaddedAndToPadded() {
        int batch = 2, maxSeqLen = 5, hiddenDim = 8;
        int[] realLens = {3, 4};

        INDArray hidden = buildPaddedHidden(batch, maxSeqLen, hiddenDim, realLens);
        INDArray mask   = buildMask(batch, maxSeqLen, realLens);

        PaddingFreeBatch pfBatch = PaddingFreeBatchCollator.fromPadded(hidden, mask);

        assertNotNull(pfBatch);
        assertEquals(batch, pfBatch.getBatchSize());
        assertEquals(4, pfBatch.getMaxSeqlenInBatch());
        assertEquals(7L, pfBatch.getUnpadded().size(0), "total_tokens should be 7");
        assertEquals((long) hiddenDim, pfBatch.getUnpadded().size(1));
        assertEquals(3, pfBatch.getCuSeqlens().size(0));

        INDArray recovered = PaddingFreeBatchCollator.toPadded(pfBatch);

        assertEquals(3, recovered.rank());
        assertEquals((long) batch, recovered.size(0));
        assertEquals((long) maxSeqLen, recovered.size(1));
        assertEquals((long) hiddenDim, recovered.size(2));

        // Real token positions match
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < realLens[b]; s++) {
                for (int d = 0; d < hiddenDim; d++) {
                    assertEquals(hidden.getFloat(new int[]{b, s, d}),
                                 recovered.getFloat(new int[]{b, s, d}),
                                 1e-5f,
                                 "Mismatch at [" + b + "," + s + "," + d + "]");
                }
            }
        }

        // Padding positions are 0
        for (int b = 0; b < batch; b++) {
            for (int s = realLens[b]; s < maxSeqLen; s++) {
                for (int d = 0; d < hiddenDim; d++) {
                    assertEquals(0f, recovered.getFloat(new int[]{b, s, d}), 1e-6f,
                            "Padding [" + b + "," + s + "," + d + "] should be 0");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // testPadInputBackward: verify pad_input_bp gathers grads correctly
    // ------------------------------------------------------------------
    @Test
    public void testPadInputBackward() {
        int batch = 2, maxSeqLen = 4, hiddenDim = 3;
        int[] realLens = {2, 3};
        long totalTokens = 5;

        INDArray hidden = buildPaddedHidden(batch, maxSeqLen, hiddenDim, realLens);
        INDArray mask   = buildMask(batch, maxSeqLen, realLens);

        // Get indices from unpad
        INDArray[] unpadOutputs = Nd4j.exec(new UnpadInput(hidden, mask));
        INDArray cuSeqlens = unpadOutputs[2];
        long actualTotal = cuSeqlens.getLong(batch);
        assertEquals(totalTokens, actualTotal);

        INDArray indicesTrimmed = actualTotal < unpadOutputs[1].size(0)
                ? unpadOutputs[1].get(NDArrayIndex.interval(0, actualTotal))
                : unpadOutputs[1];

        // Build a gradient padded tensor with known values
        INDArray gradPadded = Nd4j.zeros(DataType.FLOAT, batch, maxSeqLen, hiddenDim);
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < realLens[b]; s++) {
                for (int d = 0; d < hiddenDim; d++) {
                    gradPadded.putScalar(new int[]{b, s, d}, (b + 1) * 10f + s + d * 0.1f);
                }
            }
        }

        INDArray[] bpOutputs = Nd4j.exec(new PadInputBp(gradPadded, indicesTrimmed));
        INDArray gradUnpadded = bpOutputs[0];

        assertEquals(2, gradUnpadded.rank());
        assertEquals(totalTokens, gradUnpadded.size(0));
        assertEquals((long) hiddenDim, gradUnpadded.size(1));

        // Verify: token 0 = grad_padded[0,0,:], token 2 = grad_padded[1,0,:]
        for (int d = 0; d < hiddenDim; d++) {
            assertEquals(gradPadded.getFloat(new int[]{0, 0, d}),
                         gradUnpadded.getFloat(new long[]{0, d}), 1e-5f,
                         "grad token 0 dim " + d);
            assertEquals(gradPadded.getFloat(new int[]{1, 0, d}),
                         gradUnpadded.getFloat(new long[]{2, d}), 1e-5f,
                         "grad token 2 dim " + d);
        }
    }
}
