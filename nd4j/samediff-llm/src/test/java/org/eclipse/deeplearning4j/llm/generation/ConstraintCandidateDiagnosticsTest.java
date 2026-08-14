/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class ConstraintCandidateDiagnosticsTest {

    @Test
    void disabledDiagnosticsDoNotInspectCandidates() {
        ConstraintCandidateDiagnostics diagnostics =
                ConstraintCandidateDiagnostics.configuredForTest(false, 3, "", 10, 80);

        assertNull(diagnostics.capture(
                "prefix", new float[]{1.0f}, new float[]{1.0f}, true,
                token -> {
                    throw new AssertionError("disabled diagnostics rendered a token");
                }));
    }

    @Test
    void reportsRawAndPostConstraintRankingsWithoutChangingArrays() {
        ConstraintCandidateDiagnostics diagnostics =
                ConstraintCandidateDiagnostics.configuredForTest(
                        true, 3, "Alex Rivera", 10, 80);
        float[] raw = {1.0f, 9.0f, 7.0f, 8.0f};
        float[] masked = {
                Float.NEGATIVE_INFINITY,
                Float.NEGATIVE_INFINITY,
                7.0f,
                8.0f
        };
        float[] rawBefore = raw.clone();
        float[] maskedBefore = masked.clone();

        ConstraintCandidateDiagnostics.Snapshot snapshot = diagnostics.capture(
                "...Alex Rivera...", raw, masked, true, token -> "token-" + token);

        assertEquals(1, snapshot.event());
        assertEquals(4, snapshot.vocabularySize());
        assertEquals(2, snapshot.allowedFinite());
        assertEquals(1, snapshot.rawTop().get(0).tokenId());
        assertEquals(3, snapshot.allowedTop().get(0).tokenId());
        assertEquals("token-3", snapshot.allowedTop().get(0).text());
        org.junit.jupiter.api.Assertions.assertArrayEquals(rawBefore, raw);
        org.junit.jupiter.api.Assertions.assertArrayEquals(maskedBefore, masked);
    }

    @Test
    void contextFilterAndEventLimitBoundDiagnosticWork() {
        ConstraintCandidateDiagnostics diagnostics =
                ConstraintCandidateDiagnostics.configuredForTest(
                        true, 2, "second entity", 1, 80);
        float[] logits = {1.0f, 2.0f};

        assertNull(diagnostics.capture(
                "first entity", logits, logits, false, token -> "token-" + token));
        ConstraintCandidateDiagnostics.Snapshot snapshot = diagnostics.capture(
                "second entity", logits, logits, false, token -> "token-" + token);
        assertEquals(1, snapshot.event());
        assertNull(diagnostics.capture(
                "second entity again", logits, logits, false, token -> "token-" + token));
    }
}
