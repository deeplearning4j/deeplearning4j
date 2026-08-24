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

package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for the Java-side iArg packing contract used by the native
 * autoregressive decode op.
 */
public class TestAutoregressiveDecodeIArgs {

    private static AutoregressiveDecode newOp(int attnMaskReformatExtIdx, Set<Integer> stopIds) {
        INDArray prefillEmbeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 4);
        INDArray embeddingTable = Nd4j.zeros(DataType.FLOAT, 8, 4);
        INDArray inputIds = Nd4j.zeros(DataType.INT64, 1, 1);
        INDArray attentionMask = Nd4j.zeros(DataType.INT64, 1, 6);
        INDArray positionIds = Nd4j.zeros(DataType.INT64, 1, 1);
        INDArray[] staticKvBuffers = new INDArray[]{
                Nd4j.zeros(DataType.FLOAT, 1, 2, 6, 4),
                Nd4j.zeros(DataType.FLOAT, 1, 2, 6, 4)
        };

        return new AutoregressiveDecode(
                prefillEmbeddings,
                embeddingTable,
                inputIds,
                attentionMask,
                positionIds,
                staticKvBuffers,
                null,
                null,
                10,
                5,
                1,
                2,
                3,
                4,
                5,
                6,
                attnMaskReformatExtIdx,
                -1,  // cachePositionExtIdx — disabled for this test
                new int[]{101, 102},
                new int[]{201, 202},
                7,
                99,
                1,
                11,
                0.0,
                0,
                0.0,
                1.0,  // repetitionPenalty (1.0 = disabled)
                stopIds);
    }

    @Test
    @DisplayName("AutoregressiveDecode iArgs omit attn_mask_reformat slot when absent")
    public void testIArgsWithoutAttnMaskReformat() {
        AutoregressiveDecode op = newOp(-1, Set.of(777));
        long[] iArgs = op.iArgs();

        assertEquals(7L, iArgs[4], "optionalMask should include mask, posIds, and KV only");
        assertArrayEquals(new long[]{101, 102, 201, 202, 777},
                java.util.Arrays.copyOfRange(iArgs, 17, 22),
                "KV indices must start at iArg[17] when attn_mask_reformat is absent");
    }

    @Test
    @DisplayName("AutoregressiveDecode iArgs include attn_mask_reformat slot when present")
    public void testIArgsWithAttnMaskReformat() {
        AutoregressiveDecode op = newOp(77, Set.of());
        long[] iArgs = op.iArgs();

        assertEquals(15L, iArgs[4], "optionalMask should include attn_mask_reformat when present");
        assertArrayEquals(new long[]{77, 101, 102, 201, 202},
                java.util.Arrays.copyOfRange(iArgs, 17, 22),
                "attn_mask_reformat index must occupy iArg[17] when present");
    }

    @Test
    @DisplayName("AutoregressiveDecode appends a versioned multi-token stop trailer")
    public void testMultiTokenStopTrailer() {
        AutoregressiveDecode op = newOp(-1, Set.of(777));
        long[] legacy = op.iArgs().clone();

        op.withStopSequences(List.of(new int[]{10, 11}, new int[]{20, 21, 22}),
                new int[]{90, 91, 92});
        long[] packed = op.iArgs();

        long[] expectedPrefix = legacy.clone();
        expectedPrefix[4] |= 512L;
        assertArrayEquals(expectedPrefix, java.util.Arrays.copyOf(packed, legacy.length),
                "only the optional-mask trailer bit may change in the legacy prefix");
        assertTrue((packed[4] & 512L) != 0L, "optional mask must advertise the trailer");
        assertArrayEquals(new long[]{-1398034256L, 1L, 1L, 2L, 2L, 10L, 11L,
                        3L, 20L, 21L, 22L, 2L, 91L, 92L, 15L},
                java.util.Arrays.copyOfRange(packed, legacy.length, packed.length));
    }

    @Test
    @DisplayName("AutoregressiveDecode validates stop history before mutating iArgs")
    public void testInvalidStopHistoryLeavesIArgsUnchanged() {
        AutoregressiveDecode op = newOp(-1, Set.of(777));
        long[] before = op.iArgs().clone();

        assertThrows(IllegalArgumentException.class, () ->
                op.withStopSequences(List.of(new int[]{10, 11}), new int[]{4, -1}));
        assertArrayEquals(before, op.iArgs());
    }

    @Test
    @DisplayName("AutoregressiveDecode exposes scalar token-sampling policy defaults")
    public void testScalarTokenSamplingPolicyDefaults() {
        AutoregressiveDecode op = newOp(-1, Set.of());

        assertEquals(AutoregressiveDecode.DECODE_STRATEGY_GREEDY, op.getDecodeStrategy());
        assertEquals(1, op.getBatchMax());
        assertEquals(1, op.getWindowMax());
        assertEquals(1, op.getActiveBatch());
        assertEquals(1, op.getActiveWindow());
        assertEquals(-1, op.getHiddenOutputIdx());
        assertEquals(1, op.getNumBeams());
        assertEquals(0.0, op.getMinP(), 1e-9);
        assertEquals(0.0, op.getFrequencyPenalty(), 1e-9);
        assertEquals(0.0, op.getPresencePenalty(), 1e-9);
        assertEquals(0, op.getMinNewTokens());
        assertEquals(0, op.getGeneratedTokenOffset());
        assertEquals(0L, op.getSeed());
    }

    @Test
    @DisplayName("AutoregressiveDecode policy override does not disturb iArg packing")
    public void testDecodePolicyOverridePreservesIArgs() {
        AutoregressiveDecode op = newOp(77, Set.of(888));
        long[] before = op.iArgs().clone();

        long seed = 0x12345678ABCDEFL;
        op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_CONTRASTIVE,
                        1, 4, 1, 4, 12, 1, 1.0, 0.6, 4)
                .withSamplingPolicy(0.05, 0.2, 0.3, 6, 2, seed);

        assertArrayEquals(before, op.iArgs(), "policy metadata must stay in tArgs, not iArgs");
        assertEquals(AutoregressiveDecode.DECODE_STRATEGY_CONTRASTIVE, op.getDecodeStrategy());
        assertEquals(4, op.getWindowMax());
        assertEquals(4, op.getActiveWindow());
        assertEquals(12, op.getHiddenOutputIdx());
        assertEquals(4, op.getContrastiveTopK());
        assertEquals(0.05, op.getMinP(), 1e-9);
        assertEquals(0.2, op.getFrequencyPenalty(), 1e-9);
        assertEquals(0.3, op.getPresencePenalty(), 1e-9);
        assertEquals(6, op.getMinNewTokens());
        assertEquals(2, op.getGeneratedTokenOffset());
        assertEquals(seed, op.getSeed());
    }

    // ──────────────────────────────────────────────────────────────────
    // Additive: typical-p and XTC tArgs packing (indices 21, 22, 23)
    // ──────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("AutoregressiveDecode tArgs default typical-p and XTC values")
    public void testDefaultTypicalPAndXtcTArgs() {
        AutoregressiveDecode op = newOp(-1, Set.of());

        double[] tArgs = op.tArgs();
        assertEquals(24, tArgs.length, "disabled native repetition policy must not change the legacy tArgs ABI");
        // tArgs[21] = typicalP (1.0 = off)
        // tArgs[22] = xtcProbability (0.0 = off)
        // tArgs[23] = xtcThreshold (0.1 default)
        assertTrue(tArgs.length >= 24,
                "tArgs must contain at least 24 entries (0..23); got " + tArgs.length);
        assertEquals(1.0,  tArgs[21], 1e-9, "tArgs[21] (typicalP) default must be 1.0");
        assertEquals(0.0,  tArgs[22], 1e-9, "tArgs[22] (xtcProbability) default must be 0.0");
        assertEquals(0.1,  tArgs[23], 1e-9, "tArgs[23] (xtcThreshold) default must be 0.1");

        // Verify field getters match
        assertEquals(1.0,  op.getTypicalP(), 1e-9);
        assertEquals(0.0,  op.getXtcProbability(), 1e-9);
        assertEquals(0.1,  op.getXtcThreshold(), 1e-9);
    }

    @Test
    @DisplayName("AutoregressiveDecode withTypicalPAndXtc updates tArgs[21..23] without disturbing iArgs")
    public void testWithTypicalPAndXtcUpdatesOnlyTArgs() {
        AutoregressiveDecode op = newOp(77, Set.of(888));
        long[] iArgsBefore = op.iArgs().clone();

        op.withTypicalPAndXtc(0.9, 0.5, 0.15);

        // iArgs must be completely untouched
        assertArrayEquals(iArgsBefore, op.iArgs(),
                "withTypicalPAndXtc must not disturb iArgs");

        // tArgs[21..23] must reflect the new values
        assertEquals(0.9,  op.tArgs()[21], 1e-9, "tArgs[21] must reflect typicalP=0.9");
        assertEquals(0.5,  op.tArgs()[22], 1e-9, "tArgs[22] must reflect xtcProbability=0.5");
        assertEquals(0.15, op.tArgs()[23], 1e-9, "tArgs[23] must reflect xtcThreshold=0.15");
        assertEquals(0.9,  op.getTypicalP(), 1e-9);
        assertEquals(0.5,  op.getXtcProbability(), 1e-9);
        assertEquals(0.15, op.getXtcThreshold(), 1e-9);
    }

    @Test
    @DisplayName("AutoregressiveDecode tArgs[0..20] are not disturbed when withTypicalPAndXtc is called")
    public void testExistingTArgIndicesUnchangedByTypicalXtcUpdate() {
        AutoregressiveDecode op = newOp(-1, Set.of());
        double[] tArgsBefore = op.tArgs().clone();

        op.withTypicalPAndXtc(0.8, 0.3, 0.2);

        double[] tArgsAfter = op.tArgs();
        // Indices 0..20 must be unchanged
        for (int i = 0; i <= 20; i++) {
            assertEquals(tArgsBefore[i], tArgsAfter[i], 1e-12,
                    "tArgs[" + i + "] must not change after withTypicalPAndXtc");
        }
    }

    @Test
    @DisplayName("AutoregressiveDecode configureFromArguments restores typicalP and XTC from tArgs")
    public void testConfigureFromArgumentsRestoresTypicalXtc() {
        AutoregressiveDecode op = newOp(-1, Set.of());
        op.withTypicalPAndXtc(0.75, 0.4, 0.12);

        // Round-trip via configureFromArguments
        op.configureFromArguments();
        assertEquals(0.75, op.getTypicalP(), 1e-9, "configureFromArguments must restore typicalP");
        assertEquals(0.4,  op.getXtcProbability(), 1e-9, "configureFromArguments must restore xtcProbability");
        assertEquals(0.12, op.getXtcThreshold(), 1e-9, "configureFromArguments must restore xtcThreshold");
    }

    @Test
    @DisplayName("AutoregressiveDecode appends explicit native repetition policy at tArgs[43..44]")
    public void testNativeRepetitionPolicyPackingAndHistory() {
        AutoregressiveDecode op = newOp(-1, Set.of());
        op.withNativeRepetitionLoopTermination(6, 4);

        assertEquals(45, op.tArgs().length);
        assertEquals(-1.0, op.tArgs()[26], 1e-9,
                "padding to the additive extension must preserve disabled actual-sequence index");
        assertEquals(6.0, op.tArgs()[43], 1e-9);
        assertEquals(4.0, op.tArgs()[44], 1e-9);
        op.withSamplingPolicy(0.0, 0.0, 0.0, 0, 0, 0L);
        assertEquals(6.0, op.tArgs()[43], 1e-9,
                "policy rebuild must preserve additive repetition metadata");
        assertEquals(4.0, op.tArgs()[44], 1e-9);

        int[] history = new int[30];
        for (int i = 0; i < history.length; i++) history[i] = i;
        op.withStopSequences(List.of(), history);
        long[] packed = op.iArgs();
        assertTrue((packed[4] & 512L) != 0L);
        // A 6*4 detector needs at most 23 preceding tokens across continuation boundaries.
        assertEquals(23L, packed[packed.length - 25]);

        op.configureFromArguments();
        assertEquals(6, op.getNativeRepetitionLoopMaxPeriod());
        assertEquals(4, op.getNativeRepetitionLoopMaxRepeats());
    }

    @Test
    public void testNativeRepetitionPolicyRejectsPartialEnablement() {
        AutoregressiveDecode op = newOp(-1, Set.of());
        assertThrows(IllegalArgumentException.class,
                () -> op.withNativeRepetitionLoopTermination(6, 0));
    }

    @Test
    public void testNativeFinishReasonPreservesTimingOutputShape() {
        AutoregressiveDecode op = newOp(-1, Set.of());
        List<DataBuffer> shapes = Nd4j.getExecutioner().calculateOutputShape(op);
        assertEquals(3, shapes.size());
        assertArrayEquals(new long[]{10}, Shape.shape(shapes.get(2).asLong()));
    }
}
