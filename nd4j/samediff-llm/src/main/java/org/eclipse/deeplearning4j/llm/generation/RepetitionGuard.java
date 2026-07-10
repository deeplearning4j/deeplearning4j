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

import lombok.Getter;

import java.util.List;

/**
 * Degenerate-loop ("thinking trap") detector for unattended continuation.
 *
 * <p>Small language models sometimes fall into a repetition loop that never emits EOS — e.g. a
 * fixed short cycle of tokens repeated indefinitely. Because a continuation loop is hard-bounded by
 * the KV buffer capacity it can never run forever, but without early detection it would waste the
 * entire remaining context window producing garbage. This guard lets
 * {@link GenerationPipeline.GenerationSession#continueToCompletion(int)} stop early and report
 * {@link GenerationResult.FinishReason#REPETITION}.</p>
 *
 * <p><strong>Scope:</strong> this guard is applied <em>only</em> by the {@code continueToCompletion}
 * convenience loop — it inspects already-generated tokens to decide whether to keep looping. It never
 * runs inside the numerically-pure {@code generate} / {@code continueGeneration} primitives, so it
 * cannot perturb the "continue == single-shot" contract.</p>
 *
 * <p>Detection: the accumulated token sequence is scanned for a periodic tail — if the last
 * {@code period * maxRepeats} tokens are a repetition of a block of length {@code period}
 * (for any {@code period} in {@code 1..repeatNgram}) repeated at least {@code maxRepeats} times,
 * the sequence is considered degenerate.</p>
 */
@Getter
public final class RepetitionGuard {

    private final boolean enabled;
    /** Maximum cycle length (n-gram size) to test for, inclusive. */
    private final int repeatNgram;
    /** Number of consecutive repeats of a cycle that trips the guard. */
    private final int maxRepeats;

    public RepetitionGuard(boolean enabled, int repeatNgram, int maxRepeats) {
        this.enabled = enabled;
        this.repeatNgram = Math.max(1, repeatNgram);
        this.maxRepeats = Math.max(2, maxRepeats);
    }

    /**
     * Default guard, enabled. Catches cycles of length 1..4 repeated at least 3 times — covers the
     * common "collapse into a short N-token loop" failure mode (including single-token loops).
     */
    public static RepetitionGuard defaultOn() {
        return new RepetitionGuard(true, 4, 3);
    }

    /** A disabled guard — {@link #isDegenerate(List)} always returns {@code false}. */
    public static RepetitionGuard disabled() {
        return new RepetitionGuard(false, 0, 0);
    }

    /**
     * @param tokens the full accumulated generated-token sequence
     * @return {@code true} if the tail of {@code tokens} is a degenerate periodic repetition
     */
    public boolean isDegenerate(List<Integer> tokens) {
        if (!enabled || tokens == null) return false;
        int n = tokens.size();
        for (int period = 1; period <= repeatNgram; period++) {
            int need = period * maxRepeats;
            if (n < need) continue;
            boolean periodic = true;
            // Compare the last `need` tokens against themselves shifted by `period`.
            for (int i = n - need; i < n - period; i++) {
                if (!tokens.get(i).equals(tokens.get(i + period))) {
                    periodic = false;
                    break;
                }
            }
            if (periodic) return true;
        }
        return false;
    }
}
