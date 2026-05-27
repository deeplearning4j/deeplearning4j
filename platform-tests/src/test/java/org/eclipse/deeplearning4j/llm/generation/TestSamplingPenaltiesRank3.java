package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SamplingPenalties;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for sampling penalties with rank-3 logits [batch, seqLen, vocabSize].
 *
 * This is a regression test for a bug where applyLogitPenalties treated rank-3
 * logits as rank-2, reading sizeAt(1) (seqLen=1) as vocabSize and using
 * strides[1] (=vocabSize) as the element stride. This caused out-of-bounds
 * writes that corrupted GPU memory, producing incoherent LLM output.
 *
 * The bug manifested as: short prompts ("Hello") produced coherent responses
 * (EOS hit before corruption mattered), while longer prompts ("What is the
 * capital of France?") produced repetitive garbage that never hit EOS.
 */
@Slf4j
public class TestSamplingPenaltiesRank3 {

    private static final int VOCAB_SIZE = 1000;

    // ─── Rank-3 repetition penalty ─────────────────────────────────────

    @Test
    @DisplayName("Repetition penalty correctly penalizes tokens in rank-3 logits [1,1,vocab]")
    public void testRepetitionPenaltyRank3() {
        // Logits: [1, 1, vocabSize] — the shape produced by the decoder plan during decode
        INDArray logits = Nd4j.ones(DataType.FLOAT, 1, 1, VOCAB_SIZE).muli(5.0f);

        // Previously generated tokens: [3] — tokens 10, 20, 30 appeared before
        INDArray inputIds = Nd4j.createFromArray(new long[]{10, 20, 30});

        double repPenalty = 2.0;

        // Save original values at the penalized positions
        float origAt10 = logits.getFloat(0, 0, 10);
        float origAt20 = logits.getFloat(0, 0, 20);
        float origAt30 = logits.getFloat(0, 0, 30);
        float origAt0 = logits.getFloat(0, 0, 0);   // not penalized
        float origAt999 = logits.getFloat(0, 0, 999); // not penalized

        assertEquals(5.0f, origAt10, 1e-5f, "Precondition: all logits start at 5.0");

        // Execute — op is NOT in-place, results go to output array
        INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds, repPenalty, 0.0, 0.0, 0.0));
        INDArray output = results[0];

        // Penalized tokens: positive logits should be divided by repPenalty
        float expected = 5.0f / 2.0f; // = 2.5
        assertEquals(expected, output.getFloat(0, 0, 10), 1e-4f,
                "Token 10 should be penalized (5.0 / 2.0 = 2.5)");
        assertEquals(expected, output.getFloat(0, 0, 20), 1e-4f,
                "Token 20 should be penalized");
        assertEquals(expected, output.getFloat(0, 0, 30), 1e-4f,
                "Token 30 should be penalized");

        // Unpenalized tokens should be unchanged
        assertEquals(5.0f, output.getFloat(0, 0, 0), 1e-5f,
                "Token 0 should be unchanged (not in inputIds)");
        assertEquals(5.0f, output.getFloat(0, 0, 999), 1e-5f,
                "Token 999 should be unchanged (not in inputIds)");
    }

    @Test
    @DisplayName("Repetition penalty on negative logits in rank-3 multiplies instead of divides")
    public void testRepetitionPenaltyNegativeLogitsRank3() {
        INDArray logits = Nd4j.ones(DataType.FLOAT, 1, 1, VOCAB_SIZE).muli(-3.0f);
        INDArray inputIds = Nd4j.createFromArray(new long[]{50});

        double repPenalty = 2.0;
        INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds, repPenalty, 0.0, 0.0, 0.0));
        INDArray output = results[0];

        // Negative logit * repPenalty makes it more negative (stronger penalty)
        float expected = -3.0f * 2.0f; // = -6.0
        assertEquals(expected, output.getFloat(0, 0, 50), 1e-4f,
                "Negative logit should be multiplied by repPenalty");
        assertEquals(-3.0f, output.getFloat(0, 0, 0), 1e-5f,
                "Unpenalized token should be unchanged");
    }

    // ─── Rank-3 frequency penalty ──────────────────────────────────────

    @Test
    @DisplayName("Frequency penalty scales with token count in rank-3 logits")
    public void testFrequencyPenaltyRank3() {
        INDArray logits = Nd4j.ones(DataType.FLOAT, 1, 1, VOCAB_SIZE).muli(10.0f);

        // Token 42 appears 3 times, token 7 appears once
        INDArray inputIds = Nd4j.createFromArray(new long[]{42, 7, 42, 42});

        double freqPenalty = 1.5;
        INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds, 1.0, freqPenalty, 0.0, 0.0));
        INDArray output = results[0];

        assertEquals(10.0f - 3 * 1.5f, output.getFloat(0, 0, 42), 1e-4f,
                "Token 42 (count=3) should get 3 * freqPenalty subtracted");
        assertEquals(10.0f - 1 * 1.5f, output.getFloat(0, 0, 7), 1e-4f,
                "Token 7 (count=1) should get 1 * freqPenalty subtracted");
        assertEquals(10.0f, output.getFloat(0, 0, 0), 1e-5f,
                "Unpenalized token should be unchanged");
    }

    // ─── Rank-3 presence penalty ───────────────────────────────────────

    @Test
    @DisplayName("Presence penalty applies flat penalty in rank-3 logits")
    public void testPresencePenaltyRank3() {
        INDArray logits = Nd4j.ones(DataType.FLOAT, 1, 1, VOCAB_SIZE).muli(8.0f);

        // Token 100 appears 5 times — presence penalty is count-independent
        INDArray inputIds = Nd4j.createFromArray(new long[]{100, 100, 100, 100, 100});

        double presPenalty = 3.0;
        INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds, 1.0, 0.0, presPenalty, 0.0));
        INDArray output = results[0];

        assertEquals(8.0f - 3.0f, output.getFloat(0, 0, 100), 1e-4f,
                "Presence penalty should subtract flat amount regardless of count");
        assertEquals(8.0f, output.getFloat(0, 0, 0), 1e-5f,
                "Unpenalized token should be unchanged");
    }

    // ─── Rank-2 still works ────────────────────────────────────────────

    @Test
    @DisplayName("Repetition penalty still works correctly on rank-2 logits [1, vocab]")
    public void testRepetitionPenaltyRank2() {
        INDArray logits = Nd4j.ones(DataType.FLOAT, 1, VOCAB_SIZE).muli(5.0f);
        INDArray inputIds = Nd4j.createFromArray(new long[]{10, 20});

        INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds, 2.0, 0.0, 0.0, 0.0));
        INDArray output = results[0];

        assertEquals(2.5f, output.getFloat(0, 10), 1e-4f);
        assertEquals(2.5f, output.getFloat(0, 20), 1e-4f);
        assertEquals(5.0f, output.getFloat(0, 0), 1e-5f);
    }

    // ─── Rank-3 with seqLen > 1 ────────────────────────────────────────

    @Test
    @DisplayName("Rank-3 logits [1, seqLen, vocab] with seqLen > 1: penalty applies only to last position")
    public void testRank3PenaltyAppliesOnlyToLastSeqPosition() {
        int seqLen = 4;
        INDArray logits = Nd4j.ones(DataType.FLOAT, 1, seqLen, VOCAB_SIZE).muli(5.0f);
        INDArray inputIds = Nd4j.createFromArray(new long[]{10});

        INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds, 2.0, 0.0, 0.0, 0.0));
        INDArray output = results[0];

        // Earlier sequence positions should be UNCHANGED
        for (int s = 0; s < seqLen - 1; s++) {
            assertEquals(5.0f, output.getFloat(0, s, 10), 1e-5f,
                    "Position " + s + " should be unchanged (only last position gets penalties)");
        }

        // Last position should be penalized
        assertEquals(2.5f, output.getFloat(0, seqLen - 1, 10), 1e-4f,
                "Last position should have penalty applied");
        assertEquals(5.0f, output.getFloat(0, seqLen - 1, 0), 1e-5f,
                "Unpenalized token at last position should be unchanged");
    }

    // ─── Corruption regression test ────────────────────────────────────

    @Test
    @DisplayName("REGRESSION: rank-3 penalty does NOT corrupt memory beyond logits buffer")
    public void testRank3PenaltyDoesNotCorruptAdjacentMemory() {
        // This is the exact scenario that caused the LFM bug.
        // With the old code, rank-3 logits [1, 1, vocab] were treated as [1, vocab]
        // where vocab was read as sizeAt(1)=1, and elemStride was strides[1]=vocabSize.
        // For any tokenId > 0, the penalty wrote to offset = tokenId * vocabSize,
        // which is far out of bounds.

        int vocabSize = 151936; // Qwen/LFM typical vocab size

        // Allocate logits + a sentinel buffer immediately after
        INDArray logits = Nd4j.ones(DataType.FLOAT, 1, 1, vocabSize).muli(5.0f);
        INDArray sentinel = Nd4j.ones(DataType.FLOAT, 1, 1, vocabSize).muli(42.0f);

        // Token IDs that previously generated — these would cause OOB writes
        // Token 1 would write at offset 1 * 151936 = 151936 (already OOB by vocabSize)
        INDArray inputIds = Nd4j.createFromArray(new long[]{1, 2, 3, 100, 500});

        INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds, 1.5, 0.0, 0.0, 0.0));
        INDArray output = results[0];

        // Verify penalized tokens are correct
        assertEquals(5.0f / 1.5f, output.getFloat(0, 0, 1), 1e-3f, "Token 1 penalized");
        assertEquals(5.0f / 1.5f, output.getFloat(0, 0, 100), 1e-3f, "Token 100 penalized");
        assertEquals(5.0f / 1.5f, output.getFloat(0, 0, 500), 1e-3f, "Token 500 penalized");

        // Unpenalized tokens unchanged
        assertEquals(5.0f, output.getFloat(0, 0, 0), 1e-5f, "Token 0 unchanged");
        assertEquals(5.0f, output.getFloat(0, 0, 999), 1e-5f, "Token 999 unchanged");

        // Sentinel buffer should be completely untouched (42.0 everywhere)
        float sentinelMin = sentinel.minNumber().floatValue();
        float sentinelMax = sentinel.maxNumber().floatValue();
        assertEquals(42.0f, sentinelMin, 1e-5f, "Sentinel min should be 42.0 (no corruption)");
        assertEquals(42.0f, sentinelMax, 1e-5f, "Sentinel max should be 42.0 (no corruption)");
    }

    // ─── Simulated decode loop: repetitive text detection ──────────────

    @Test
    @DisplayName("REGRESSION: simulated decode loop with penalties does not produce degenerate repetition")
    public void testDecodeLoopWithPenaltiesProducesVariedTokens() {
        // Simulate the autoregressive decode loop that was producing
        // "Paris is London is London is..." with LFM.
        //
        // Setup: vocab of 100, all logits equal. With repetition penalty,
        // the sampler should avoid repeating tokens. Without the rank-3 fix,
        // penalties were applied to wrong memory and had no effect on the
        // actual logits, causing the model to repeat whatever the initial
        // bias was.

        int vocabSize = 100;
        int maxSteps = 30;
        double repPenalty = 1.5;

        int[] generated = new int[maxSteps];

        for (int step = 0; step < maxSteps; step++) {
            // Simulate plan output: rank-3 logits [1, 1, vocab]
            // All equal to 1.0, so without penalty any token is equally likely
            // but with penalty, previously-seen tokens get suppressed
            INDArray logits = Nd4j.ones(DataType.FLOAT, 1, 1, vocabSize);

            INDArray penalized;
            if (step > 0) {
                // Build inputIds from tokens generated so far
                long[] tokensSoFar = new long[step];
                for (int i = 0; i < step; i++) {
                    tokensSoFar[i] = generated[i];
                }
                INDArray inputIds = Nd4j.createFromArray(tokensSoFar);
                INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds, repPenalty, 0.0, 0.0, 0.0));
                penalized = results[0];
            } else {
                penalized = logits;
            }

            // Greedy: pick argmax of last position
            INDArray lastPos = penalized.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            generated[step] = lastPos.argMax(0).getInt(0);

            logits.close();
            if (penalized != logits) penalized.close();
        }

        // Count unique tokens — with repetition penalty on uniform logits,
        // greedy decode should cycle through many distinct tokens, not repeat
        Set<Integer> unique = new HashSet<>();
        for (int tok : generated) unique.add(tok);

        log.info("Generated {} tokens, {} unique: {}", maxSteps, unique.size(),
                Arrays.toString(generated));

        // With penalty working correctly, we should see many distinct tokens.
        // Without the fix, all tokens would be the same (e.g., always 0)
        // because the penalty writes went OOB and never actually modified the logits.
        assertTrue(unique.size() >= maxSteps / 2,
                "Expected at least " + (maxSteps / 2) + " unique tokens with rep penalty, " +
                "got " + unique.size() + ". Penalty may not be applying correctly. " +
                "Tokens: " + Arrays.toString(generated));
    }

    // ─── Combined penalties rank-3 ─────────────────────────────────────

    @Test
    @DisplayName("All three penalties combined work correctly on rank-3 logits")
    public void testCombinedPenaltiesRank3() {
        INDArray logits = Nd4j.ones(DataType.FLOAT, 1, 1, VOCAB_SIZE).muli(10.0f);

        // Token 50 appears twice, token 75 appears once
        INDArray inputIds = Nd4j.createFromArray(new long[]{50, 75, 50});

        double repPenalty = 2.0;
        double freqPenalty = 1.0;
        double presPenalty = 0.5;

        INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds,
                repPenalty, freqPenalty, presPenalty, 0.0));
        INDArray output = results[0];

        // Token 50 (count=2): 10/2 - 2*1.0 - 0.5 = 5 - 2 - 0.5 = 2.5
        assertEquals(2.5f, output.getFloat(0, 0, 50), 1e-4f,
                "Token 50 should have all three penalties applied");

        // Token 75 (count=1): 10/2 - 1*1.0 - 0.5 = 5 - 1 - 0.5 = 3.5
        assertEquals(3.5f, output.getFloat(0, 0, 75), 1e-4f,
                "Token 75 should have all three penalties applied");

        // Unpenalized
        assertEquals(10.0f, output.getFloat(0, 0, 0), 1e-5f);
    }

    // ─── FP16 rank-3 ───────────────────────────────────────────────────

    @ParameterizedTest
    @ValueSource(strings = {"FLOAT", "HALF", "BFLOAT16"})
    @DisplayName("Repetition penalty works for different dtypes on rank-3 logits")
    public void testRepetitionPenaltyDtypes(String dtypeName) {
        DataType dtype = DataType.valueOf(dtypeName);
        INDArray logits = Nd4j.ones(dtype, 1, 1, VOCAB_SIZE).muli(6.0);
        INDArray inputIds = Nd4j.createFromArray(new long[]{5});

        INDArray[] results = Nd4j.exec(new SamplingPenalties(logits, inputIds, 3.0, 0.0, 0.0, 0.0));
        INDArray output = results[0];

        float expected = 6.0f / 3.0f; // = 2.0
        float tolerance = (dtype == DataType.HALF || dtype == DataType.BFLOAT16) ? 0.1f : 1e-4f;

        assertEquals(expected, output.getFloat(0, 0, 5), tolerance,
                dtype + ": Token 5 should be penalized (6/3=2)");
        assertEquals(6.0f, output.getFloat(0, 0, 0), tolerance,
                dtype + ": Token 0 should be unchanged");
    }
}
