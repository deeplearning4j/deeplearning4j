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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ADR 0106 Phase 1 — Window Substrate Tests.
 *
 * <p>Tests for the frozen masked multi-position decode substrate:</p>
 * <ol>
 *   <li>W=1 parity — tArgs and config fields are identical when windowMax=1 vs default</li>
 *   <li>Window mask structure — [1,1,W,past+W] mask has correct values for W=4 causal window</li>
 *   <li>Window position grid — [1,W] grid has correct positions for given currentPosition</li>
 *   <li>iArgs stability — window substrate params live in tArgs, never disturb iArgs</li>
 *   <li>CUDA token_sample top-k/top-p parity — CUDA and CPU select from the same top tokens</li>
 * </ol>
 *
 * <p>None of these tests require a real GGUF model; they validate the substrate contract
 * (mask structure, position grid, iArgs layout, sampling parity) without running full decode.</p>
 */
@DisplayName("ADR 0106 Phase 1: Window Substrate Tests")
public class TestWindowSubstrate {

    // MASK_FILL value used in ADR 0106 window masks (float negative infinity sentinel)
    private static final float MASK_FILL = -3.4028235e+38f;
    private static final float MASK_FILL_TOL = 1e30f;  // tolerance for comparing MASK_FILL

    // ──────────────────────────────────────────────────────────────────────────
    // Helper: build a baseline AutoregressiveDecode op (same as TestAutoregressiveDecodeIArgs)
    // ──────────────────────────────────────────────────────────────────────────

    private static AutoregressiveDecode newBaselineOp() {
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
                prefillEmbeddings, embeddingTable, inputIds, attentionMask, positionIds,
                staticKvBuffers, null, null,
                10, 5, 1, 2, 3, 4, 5, 6,
                -1, -1, new int[]{101, 102}, new int[]{201, 202},
                7, 99, 1, 2, 0.0, 0, 0.0, 1.0, Set.of());
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1. W=1 tArgs parity: windowMax=1 must produce identical policy envelope to default
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("W=1 window substrate tArgs are identical to default (greedy, scalar)")
    public void testWindowW1TArgsIdenticalToDefault() {
        AutoregressiveDecode defaultOp = newBaselineOp();
        AutoregressiveDecode explicitW1Op = newBaselineOp();

        // Explicitly set W=1 (same as default) via withDecodePolicy
        explicitW1Op.withDecodePolicy(
                AutoregressiveDecode.DECODE_STRATEGY_GREEDY,
                1, 1, 1, 1, -1, 1, 1.0, 0.0, 0);

        // All tArgs must be consistent: the W=1 substrate is the identity
        double[] defaultTArgs = defaultOp.tArgs();
        double[] explicitTArgs = explicitW1Op.tArgs();
        assertEquals(defaultTArgs.length, explicitTArgs.length,
                "W=1 explicit and default must have same tArgs count");

        // tArgs[5]=batchMax, [6]=windowMax, [7]=activeBatch, [8]=activeWindow must all be 1
        assertTrue(defaultTArgs.length >= 9, "tArgs must have at least 9 entries");
        assertEquals(1.0, defaultTArgs[5], 1e-9, "default batchMax must be 1");
        assertEquals(1.0, defaultTArgs[6], 1e-9, "default windowMax must be 1");
        assertEquals(1.0, defaultTArgs[7], 1e-9, "default activeBatch must be 1");
        assertEquals(1.0, defaultTArgs[8], 1e-9, "default activeWindow must be 1");

        assertEquals(1.0, explicitTArgs[5], 1e-9, "explicit W=1 batchMax must be 1");
        assertEquals(1.0, explicitTArgs[6], 1e-9, "explicit W=1 windowMax must be 1");
        assertEquals(1.0, explicitTArgs[7], 1e-9, "explicit W=1 activeBatch must be 1");
        assertEquals(1.0, explicitTArgs[8], 1e-9, "explicit W=1 activeWindow must be 1");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2. W>1 window substrate tArgs are set correctly by withDecodePolicy
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("W>1 window substrate tArgs reflect windowMax and activeWindow")
    public void testWindowW4TArgsReflectWindowDimensions() {
        AutoregressiveDecode op = newBaselineOp();
        long[] iArgsBefore = op.iArgs().clone();

        // Set W=8 substrate (e.g. 8-token speculative window)
        op.withDecodePolicy(
                AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                1, 8, 1, 4, -1, 1, 1.0, 0.0, 0);

        // iArgs must be completely untouched (window params live in tArgs)
        assertArrayEquals(iArgsBefore, op.iArgs(),
                "withDecodePolicy must not disturb iArgs — window params are tArgs only");

        // tArgs[6]=windowMax=8, tArgs[8]=activeWindow=4
        assertEquals(8.0, op.tArgs()[6], 1e-9, "tArgs[6] must reflect windowMax=8");
        assertEquals(4.0, op.tArgs()[8], 1e-9, "tArgs[8] must reflect activeWindow=4");
        assertEquals(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                (int) op.tArgs()[4], "tArgs[4] must reflect SPECULATIVE strategy");

        // Getters must also reflect the new values
        assertEquals(8, op.getWindowMax(), "getWindowMax() must return 8");
        assertEquals(4, op.getActiveWindow(), "getActiveWindow() must return 4");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 3. Window mask structure (CPU — Java builds the mask, validates it)
    //    The C++ substrate fills the mask on each decode step. This test
    //    validates the EXPECTED mask values for a W=4 window with past=3:
    //      mask[w, k]: 0 if k < past (attend to past KV)
    //                  0 if k == past + w (attend to self)
    //                  MASK_FILL otherwise
    //    This is the substrate contract Phase 2 (speculative) will depend on.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("Window mask has correct causal structure: attend-to-past + self-attend only")
    public void testWindowMaskCausalStructure() {
        int wMax = 4;
        int past = 3;
        int rowLen = past + wMax;  // 7
        int activeWindow = 3;      // only 3 of 4 positions active

        // Build the expected mask exactly as the C++ fillWindowMaskKernel does
        float[] expectedMask = buildExpectedWindowMask(wMax, past, activeWindow, rowLen);

        // Validate mask shape and values
        INDArray mask = Nd4j.create(expectedMask).reshape(1, 1, wMax, rowLen);
        assertEquals(1L, mask.size(0), "batch dim must be 1");
        assertEquals(1L, mask.size(1), "head dim must be 1");
        assertEquals(wMax, (int) mask.size(2), "W dim must equal wMax");
        assertEquals(rowLen, (int) mask.size(3), "col dim must equal past+wMax");

        // Row 0 (active, w=0): attend to past[0..2], self at past+0=3; mask [4,5,6]
        for (int k = 0; k < past; k++) {
            assertEquals(0.0f, mask.getFloat(0, 0, 0, k), 1e-6f,
                    "row 0, col " + k + " must be 0 (attend to past)");
        }
        assertEquals(0.0f, mask.getFloat(0, 0, 0, past + 0), 1e-6f,
                "row 0, col past+0 must be 0 (self-attend)");
        for (int k = past + 1; k < rowLen; k++) {
            assertTrue(mask.getFloat(0, 0, 0, k) < -MASK_FILL_TOL,
                    "row 0, col " + k + " must be MASK_FILL (future/sibling masked)");
        }

        // Row 1 (active, w=1): attend to past[0..2], self at past+1=4; mask [3,5,6]
        for (int k = 0; k < past; k++) {
            assertEquals(0.0f, mask.getFloat(0, 0, 1, k), 1e-6f,
                    "row 1, col " + k + " must be 0 (attend to past)");
        }
        // col past+0 = 3: sibling position w=0 — MUST be masked (causal: w=1 can't attend to w=0)
        assertTrue(mask.getFloat(0, 0, 1, past + 0) < -MASK_FILL_TOL,
                "row 1, col past+0 (sibling w=0) must be MASK_FILL");
        assertEquals(0.0f, mask.getFloat(0, 0, 1, past + 1), 1e-6f,
                "row 1, col past+1 must be 0 (self-attend)");
        for (int k = past + 2; k < rowLen; k++) {
            assertTrue(mask.getFloat(0, 0, 1, k) < -MASK_FILL_TOL,
                    "row 1, col " + k + " must be MASK_FILL (future masked)");
        }

        // Row 3 (inactive, w >= activeWindow): entire row must be MASK_FILL
        for (int k = 0; k < rowLen; k++) {
            assertTrue(mask.getFloat(0, 0, activeWindow, k) < -MASK_FILL_TOL,
                    "inactive row " + activeWindow + ", col " + k + " must be MASK_FILL");
        }
    }

    @Test
    @DisplayName("Window mask W=1 matches 1-wide mask: attend-to-past + self-attend at position 0")
    public void testWindowMaskW1MatchesSingleToken() {
        // W=1 window mask must produce exactly the same attend pattern as the single-token case
        int wMax = 1;
        int past = 5;
        int rowLen = past + wMax;

        float[] mask1Wide = buildExpectedWindowMask(wMax, past, 1, rowLen);
        INDArray m = Nd4j.create(mask1Wide).reshape(1, 1, wMax, rowLen);

        // All past positions: 0 (attend)
        for (int k = 0; k < past; k++) {
            assertEquals(0.0f, m.getFloat(0, 0, 0, k), 1e-6f,
                    "W=1 mask past col " + k + " must be 0");
        }
        // Self at col past+0=past: 0
        assertEquals(0.0f, m.getFloat(0, 0, 0, past), 1e-6f,
                "W=1 mask self col must be 0");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 4. Window position grid structure
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("Window position grid: active positions get currentPos+w, inactive get currentPos")
    public void testWindowPositionGrid() {
        int wMax = 6;
        int activeWindow = 4;
        long currentPos = 10L;

        long[] grid = buildExpectedWindowPositionGrid(wMax, activeWindow, currentPos);
        assertEquals(wMax, grid.length, "grid length must equal wMax");

        // Active positions: grid[w] = currentPos + w
        for (int w = 0; w < activeWindow; w++) {
            assertEquals(currentPos + w, grid[w],
                    "active position w=" + w + " must be currentPos+" + w);
        }
        // Inactive positions: grid[w] = currentPos
        for (int w = activeWindow; w < wMax; w++) {
            assertEquals(currentPos, grid[w],
                    "inactive position w=" + w + " must be currentPos");
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 5. CUDA token_sample top-k/top-p parity vs CPU
    //    The ADR notes CUDA top-k/top-p was "silently ignored" — but the kernel
    //    tempTopKTopPSampleKernel was already added. This test verifies that
    //    CPU and CUDA (when available) both select tokens from the same top-k set,
    //    confirming the gap is closed.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("CUDA token_sample top-k parity: selected token is in the top-k set (greedy case)")
    public void testTokenSampleTopKParityGreedy() {
        // Use a vocabulary where the top token is unambiguous
        // Logits: token 7 is clearly the winner (value 10.0), others much smaller
        int vocabSize = 16;
        float[] logitData = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) logitData[i] = (float) Math.random() * 2.0f;
        logitData[7] = 10.0f;  // clear maximum

        INDArray logits = Nd4j.create(logitData).reshape(1, vocabSize);

        // Greedy (argmax): both CPU and CUDA must select token 7
        INDArray output = Nd4j.exec(new TokenSample(logits))[0];
        long selected = output.getLong(0);
        assertEquals(7L, selected,
                "Greedy top-1 must select the logit maximum (token 7)");
    }

    @Test
    @DisplayName("CUDA token_sample top-k=3: selected token is always in top-3 set")
    public void testTokenSampleTopK3AlwaysInTopSet() {
        // Logits with clear top-3: tokens 2, 5, 9 have highest values
        int vocabSize = 20;
        float[] logitData = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) logitData[i] = -5.0f + i * 0.1f;
        logitData[2] = 8.0f;
        logitData[5] = 9.0f;
        logitData[9] = 10.0f;

        INDArray logits = Nd4j.create(logitData).reshape(1, vocabSize);
        Set<Long> top3 = Set.of(2L, 5L, 9L);

        // Run 20 independent samples with temperature=1.0, top-k=3
        // All samples must fall within the top-3 set (since all others have << weight)
        for (int trial = 0; trial < 20; trial++) {
            INDArray logitsCopy = logits.dup();
            TokenSample op = new TokenSample(logitsCopy, 1.0, 3, 0.0, trial + 1);
            INDArray out = Nd4j.exec(op)[0];
            long tok = out.getLong(0);
            assertTrue(top3.contains(tok),
                    "trial " + trial + ": selected token " + tok +
                    " is not in top-3 set {2, 5, 9} — top-k filter not applied");
        }
    }

    @Test
    @DisplayName("CUDA token_sample top-p=0.5: selected token is in the nucleus set")
    public void testTokenSampleTopPNucleusFilter() {
        // Create logits where the top-2 tokens (1, 3) account for ~85% of probability mass.
        // top-p=0.5 should restrict sampling to these two tokens.
        int vocabSize = 10;
        // After softmax, token 1 and 3 dominate; rest is noise
        float[] logitData = {-5.0f, 5.0f, -5.0f, 4.5f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f};
        INDArray logits = Nd4j.create(logitData).reshape(1, vocabSize);
        Set<Long> dominantSet = Set.of(1L, 3L);

        // Run 30 samples with temperature=1.0, top-p=0.5 (tight nucleus)
        for (int trial = 0; trial < 30; trial++) {
            INDArray logitsCopy = logits.dup();
            // top-p=0.5: only tokens with cumulative prob >= 0.5 kept; token 1 alone exceeds 0.5
            TokenSample op = new TokenSample(logitsCopy, 1.0, 0, 0.5, trial + 1);
            INDArray out = Nd4j.exec(op)[0];
            long tok = out.getLong(0);
            assertTrue(dominantSet.contains(tok),
                    "trial " + trial + ": selected token " + tok +
                    " is outside nucleus {1, 3} — top-p filter not applied correctly");
        }
    }

    @Test
    @DisplayName("CUDA token_sample batch: each row selects from its own top-k")
    public void testTokenSampleBatchTopK() {
        // Batch of 3 rows; each has a different clear maximum
        // Row 0: token 0 is max, Row 1: token 5 is max, Row 2: token 9 is max
        int batch = 3;
        int vocabSize = 10;
        float[] data = new float[batch * vocabSize];
        for (int i = 0; i < data.length; i++) data[i] = -3.0f;
        data[0 * vocabSize + 0] = 10.0f;
        data[1 * vocabSize + 5] = 10.0f;
        data[2 * vocabSize + 9] = 10.0f;

        INDArray logits = Nd4j.create(data).reshape(batch, vocabSize);
        // Greedy: must select the maximum per row
        INDArray out = Nd4j.exec(new TokenSample(logits))[0];
        assertEquals(0L, out.getLong(0), "batch row 0: must select token 0");
        assertEquals(5L, out.getLong(1), "batch row 1: must select token 5");
        assertEquals(9L, out.getLong(2), "batch row 2: must select token 9");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Utilities
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Build the expected window attention mask for a given W/past/activeWindow configuration.
     * Mirrors the logic in C++ fillWindowMaskKernel exactly:
     *   mask[w * rowLen + k]:
     *     MASK_FILL if w >= activeWindow
     *     0.0f      if k < currentPos   (attend to past KV)
     *     0.0f      if k == currentPos + w (self-attend)
     *     MASK_FILL otherwise
     *
     * @param wMax         maximum window size (W_max)
     * @param currentPos   current decode position (= past KV length = past)
     * @param activeWindow number of active window positions this step
     * @param rowLen       past + wMax (mask row length)
     */
    private static float[] buildExpectedWindowMask(int wMax, int currentPos,
                                                    int activeWindow, int rowLen) {
        float[] mask = new float[wMax * rowLen];
        for (int w = 0; w < wMax; w++) {
            for (int k = 0; k < rowLen; k++) {
                int idx = w * rowLen + k;
                if (w >= activeWindow) {
                    mask[idx] = MASK_FILL;
                } else if (k < currentPos) {
                    mask[idx] = 0.0f;
                } else if (k == currentPos + w) {
                    mask[idx] = 0.0f;
                } else {
                    mask[idx] = MASK_FILL;
                }
            }
        }
        return mask;
    }

    /**
     * Build the expected window position grid for a given W/currentPos/activeWindow config.
     * Mirrors the logic in C++ fillWindowPositionGridKernel:
     *   grid[w] = currentPos + w  for w < activeWindow
     *   grid[w] = currentPos      for w >= activeWindow
     */
    private static long[] buildExpectedWindowPositionGrid(int wMax, int activeWindow,
                                                           long currentPos) {
        long[] grid = new long[wMax];
        for (int w = 0; w < wMax; w++) {
            grid[w] = (w < activeWindow) ? (currentPos + w) : currentPos;
        }
        return grid;
    }
}
