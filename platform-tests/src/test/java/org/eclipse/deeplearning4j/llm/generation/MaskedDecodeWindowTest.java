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

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the fixed-width masked multi-position decode window builders
 * ({@code DecoderInputBuilder.buildInGraphWindowMask} / {@code buildInGraphWindowPositionIds}),
 * ADR 0106 Phase 1 — the shared substrate mask that beam / contrastive / speculative ride on.
 *
 * <p>Pure array logic, runs on any backend (CPU used in CI).</p>
 */
class MaskedDecodeWindowTest {

    private static final float ATTEND = 0.0f;
    private static final float MASKED = -1e9f;   // FLOAT dtype-safe mask value (matches DecoderInputBuilder)

    private static float m(INDArray mask, int q, int k) {
        return mask.getFloat(0, 0, q, k);
    }

    /** W=1 window is byte-for-byte the existing single-step decode mask (the substrate is a superset). */
    @Test
    void windowW1EqualsDecodeMask() {
        long maxKv = 12;
        for (long cp = 0; cp < maxKv; cp++) {
            INDArray decode = DecoderInputBuilder.buildInGraphDecodeMask(cp, maxKv, DataType.FLOAT);
            INDArray window = DecoderInputBuilder.buildInGraphWindowMask(
                    DecoderInputBuilder.chainParents(1, 1), cp, 1, 1, maxKv, DataType.FLOAT);
            assertArrayEquals(new long[]{1, 1, 1, maxKv}, window.shape());
            for (int k = 0; k < maxKv; k++) {
                assertEquals(decode.getFloat(0, 0, 0, k), m(window, 0, k), 0.0f,
                        "W=1 window vs decode mask mismatch at cp=" + cp + " k=" + k);
            }
            decode.close();
            window.close();
        }
    }

    /** Contrastive: the wActive candidates are alternatives at one step — each attends to past + self only. */
    @Test
    void siblingWindowCandidatesAreIndependent() {
        int wMax = 4, wActive = 3;
        int cp = 5, maxKv = 16;
        INDArray mask = DecoderInputBuilder.buildInGraphWindowMask(
                DecoderInputBuilder.siblingParents(wActive, wMax), cp, wActive, wMax, maxKv, DataType.FLOAT);
        assertArrayEquals(new long[]{1, 1, wMax, maxKv}, mask.shape());
        for (int j = 0; j < wMax; j++) {
            for (int k = 0; k < cp; k++) assertEquals(ATTEND, m(mask, j, k), "past col j=" + j + " k=" + k);
            for (int a = 0; a < wMax; a++) {
                float v = m(mask, j, cp + a);
                if (a == j) assertEquals(ATTEND, v, "self col slot " + j);
                else assertEquals(MASKED, v, "sibling col a=" + a + " must be masked for slot " + j);
            }
            for (int k = cp + wMax; k < maxKv; k++) assertEquals(MASKED, m(mask, j, k), "pad col j=" + j);
        }
        mask.close();
    }

    /** Speculative chain: slot j attends to past + window slots 0..j (causal within the window). */
    @Test
    void chainWindowIsCausalWithinWindow() {
        int wMax = 5, wActive = 4;
        int cp = 3, maxKv = 16;
        INDArray mask = DecoderInputBuilder.buildInGraphWindowMask(
                DecoderInputBuilder.chainParents(wActive, wMax), cp, wActive, wMax, maxKv, DataType.FLOAT);
        for (int j = 0; j < wActive; j++) {
            for (int k = 0; k < cp; k++) assertEquals(ATTEND, m(mask, j, k), "past j=" + j);
            for (int a = 0; a < wMax; a++) {
                float v = m(mask, j, cp + a);
                if (a <= j) assertEquals(ATTEND, v, "chain ancestor a=" + a + " for slot " + j);
                else assertEquals(MASKED, v, "future window col a=" + a + " for slot " + j);
            }
        }
        mask.close();
    }

    /** Inactive slots (>= wActive) must keep a finite softmax row (past + self), never all-masked (NaN). */
    @Test
    void inactiveSlotsAreFiniteAndPastPlusSelf() {
        int wMax = 4, wActive = 2;
        int cp = 4, maxKv = 12;
        INDArray mask = DecoderInputBuilder.buildInGraphWindowMask(
                DecoderInputBuilder.chainParents(wActive, wMax), cp, wActive, wMax, maxKv, DataType.FLOAT);
        for (int j = wActive; j < wMax; j++) {
            int attendCount = 0;
            for (int k = 0; k < maxKv; k++) if (m(mask, j, k) == ATTEND) attendCount++;
            assertTrue(attendCount >= 1, "inactive slot " + j + " must have >=1 attend col (finite softmax)");
            assertEquals(ATTEND, m(mask, j, cp + j), "inactive self col slot " + j);
        }
        mask.close();
    }

    /** Position grid: chain increments from cachePos; siblings share cachePos; shape [1,wMax] LONG. */
    @Test
    void positionGridsChainAndSibling() {
        int wMax = 4, wActive = 3;
        long cp = 7;
        INDArray chain = DecoderInputBuilder.buildInGraphWindowPositionIds(
                DecoderInputBuilder.chainPositions(wActive, wMax), cp, wMax);
        INDArray sib = DecoderInputBuilder.buildInGraphWindowPositionIds(
                DecoderInputBuilder.siblingPositions(wActive, wMax), cp, wMax);
        assertArrayEquals(new long[]{1, wMax}, chain.shape());
        assertEquals(DataType.LONG, chain.dataType());
        for (int j = 0; j < wActive; j++) {
            assertEquals(cp + j, chain.getLong(0, j), "chain pos slot " + j);
            assertEquals(cp, sib.getLong(0, j), "sibling pos slot " + j);
        }
        chain.close();
        sib.close();
    }
}
