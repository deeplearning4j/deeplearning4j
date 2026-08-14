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

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntFunction;

/**
 * Opt-in observation of raw and post-constraint token rankings.
 *
 * <p>This diagnostic consumes the CPU-side arrays already created by constrained decoding. It
 * never mutates logits, the constraint automaton, sampling state, or generated tokens. Enable it
 * with {@value #ENABLED_PROPERTY}. Optional properties narrow the emitted-text context and bound
 * output volume:</p>
 *
 * <ul>
 *     <li>{@value #TOP_K_PROPERTY}: candidates per ranking (default 10, maximum 100)</li>
 *     <li>{@value #EMITTED_CONTAINS_PROPERTY}: log only after emitted text contains this value</li>
 *     <li>{@value #MAX_EVENTS_PROPERTY}: maximum matching decode steps (default 200)</li>
 *     <li>{@value #CONTEXT_CHARS_PROPERTY}: emitted-text tail length (default 320)</li>
 * </ul>
 */
@Slf4j
final class ConstraintCandidateDiagnostics {

    static final String ENABLED_PROPERTY =
            "org.eclipse.deeplearning4j.llm.constraint.candidateDiagnostics";
    static final String TOP_K_PROPERTY = ENABLED_PROPERTY + ".topK";
    static final String EMITTED_CONTAINS_PROPERTY = ENABLED_PROPERTY + ".emittedContains";
    static final String MAX_EVENTS_PROPERTY = ENABLED_PROPERTY + ".maxEvents";
    static final String CONTEXT_CHARS_PROPERTY = ENABLED_PROPERTY + ".contextChars";

    private final boolean enabled;
    private final int topK;
    private final String emittedContains;
    private final int maxEvents;
    private final int contextChars;
    private int capturedEvents;

    private ConstraintCandidateDiagnostics(
            boolean enabled,
            int topK,
            String emittedContains,
            int maxEvents,
            int contextChars) {
        this.enabled = enabled;
        this.topK = bounded(topK, 1, 100);
        this.emittedContains = emittedContains == null ? "" : emittedContains;
        this.maxEvents = bounded(maxEvents, 1, 10_000);
        this.contextChars = bounded(contextChars, 32, 4_096);
    }

    static ConstraintCandidateDiagnostics fromSystemProperties() {
        return new ConstraintCandidateDiagnostics(
                Boolean.parseBoolean(System.getProperty(ENABLED_PROPERTY, "false")),
                Integer.getInteger(TOP_K_PROPERTY, 10),
                System.getProperty(EMITTED_CONTAINS_PROPERTY, ""),
                Integer.getInteger(MAX_EVENTS_PROPERTY, 200),
                Integer.getInteger(CONTEXT_CHARS_PROPERTY, 320));
    }

    static ConstraintCandidateDiagnostics configuredForTest(
            boolean enabled,
            int topK,
            String emittedContains,
            int maxEvents,
            int contextChars) {
        return new ConstraintCandidateDiagnostics(
                enabled, topK, emittedContains, maxEvents, contextChars);
    }

    Snapshot captureAndLog(
            String emittedText,
            float[] rawLogits,
            float[] maskedLogits,
            boolean greedy,
            IntFunction<String> tokenRenderer) {
        Snapshot snapshot = capture(
                emittedText, rawLogits, maskedLogits, greedy, tokenRenderer);
        if (snapshot != null) {
            log.info("[ConstraintCandidates] event={} greedy={} emitted=\"{}\" "
                            + "rawTop={} allowedTop={} allowedFinite={}/{}",
                    snapshot.event(), snapshot.greedy(), snapshot.emittedTail(),
                    formatCandidates(snapshot.rawTop()),
                    formatCandidates(snapshot.allowedTop()),
                    snapshot.allowedFinite(), snapshot.vocabularySize());
        }
        return snapshot;
    }

    Snapshot capture(
            String emittedText,
            float[] rawLogits,
            float[] maskedLogits,
            boolean greedy,
            IntFunction<String> tokenRenderer) {
        if (!enabled || capturedEvents >= maxEvents) {
            return null;
        }
        String emitted = emittedText == null ? "" : emittedText;
        if (!emittedContains.isEmpty() && !emitted.contains(emittedContains)) {
            return null;
        }
        if (rawLogits == null || maskedLogits == null
                || rawLogits.length != maskedLogits.length) {
            throw new IllegalArgumentException(
                    "Raw and masked logits must be non-null and have equal vocabulary length");
        }
        if (tokenRenderer == null) {
            throw new IllegalArgumentException("tokenRenderer must not be null");
        }

        int event = ++capturedEvents;
        List<Candidate> rawTop = topCandidates(rawLogits, topK, tokenRenderer);
        List<Candidate> allowedTop = topCandidates(maskedLogits, topK, tokenRenderer);
        int allowedFinite = 0;
        for (float value : maskedLogits) {
            if (!Float.isNaN(value) && value > Float.NEGATIVE_INFINITY) {
                allowedFinite++;
            }
        }
        return new Snapshot(
                event,
                greedy,
                escapeTail(emitted, contextChars),
                rawTop,
                allowedTop,
                allowedFinite,
                rawLogits.length);
    }

    private static List<Candidate> topCandidates(
            float[] logits, int limit, IntFunction<String> tokenRenderer) {
        int size = Math.min(limit, logits.length);
        int[] ids = new int[size];
        float[] scores = new float[size];
        Arrays.fill(ids, -1);
        Arrays.fill(scores, Float.NEGATIVE_INFINITY);

        for (int tokenId = 0; tokenId < logits.length; tokenId++) {
            float score = logits[tokenId];
            if (Float.isNaN(score) || score == Float.NEGATIVE_INFINITY) {
                continue;
            }
            int insertion = size;
            for (int index = 0; index < size; index++) {
                if (ids[index] < 0 || score > scores[index]
                        || (score == scores[index] && tokenId < ids[index])) {
                    insertion = index;
                    break;
                }
            }
            if (insertion >= size) {
                continue;
            }
            for (int index = size - 1; index > insertion; index--) {
                ids[index] = ids[index - 1];
                scores[index] = scores[index - 1];
            }
            ids[insertion] = tokenId;
            scores[insertion] = score;
        }

        List<Candidate> result = new ArrayList<>(size);
        for (int index = 0; index < size && ids[index] >= 0; index++) {
            result.add(new Candidate(
                    ids[index], scores[index], tokenRenderer.apply(ids[index])));
        }
        return List.copyOf(result);
    }

    private static String formatCandidates(List<Candidate> candidates) {
        StringBuilder value = new StringBuilder("[");
        for (int index = 0; index < candidates.size(); index++) {
            if (index > 0) {
                value.append(", ");
            }
            Candidate candidate = candidates.get(index);
            value.append("{id=").append(candidate.tokenId())
                    .append(", logit=").append(candidate.logit())
                    .append(", text=\"").append(escapeTail(candidate.text(), 160))
                    .append("\"}");
        }
        return value.append(']').toString();
    }

    private static String escapeTail(String value, int maxChars) {
        String escaped = value == null ? "" : value
                .replace("\\", "\\\\")
                .replace("\r", "\\r")
                .replace("\n", "\\n")
                .replace("\t", "\\t")
                .replace("\"", "\\\"");
        return escaped.length() <= maxChars
                ? escaped : "…" + escaped.substring(escaped.length() - maxChars);
    }

    private static int bounded(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    static final class Candidate {
        private final int tokenId;
        private final float logit;
        private final String text;

        Candidate(int tokenId, float logit, String text) {
            this.tokenId = tokenId;
            this.logit = logit;
            this.text = text;
        }

        int tokenId() {
            return tokenId;
        }

        float logit() {
            return logit;
        }

        String text() {
            return text;
        }
    }

    static final class Snapshot {
        private final int event;
        private final boolean greedy;
        private final String emittedTail;
        private final List<Candidate> rawTop;
        private final List<Candidate> allowedTop;
        private final int allowedFinite;
        private final int vocabularySize;

        Snapshot(
                int event,
                boolean greedy,
                String emittedTail,
                List<Candidate> rawTop,
                List<Candidate> allowedTop,
                int allowedFinite,
                int vocabularySize) {
            this.event = event;
            this.greedy = greedy;
            this.emittedTail = emittedTail;
            this.rawTop = rawTop;
            this.allowedTop = allowedTop;
            this.allowedFinite = allowedFinite;
            this.vocabularySize = vocabularySize;
        }

        int event() {
            return event;
        }

        boolean greedy() {
            return greedy;
        }

        String emittedTail() {
            return emittedTail;
        }

        List<Candidate> rawTop() {
            return rawTop;
        }

        List<Candidate> allowedTop() {
            return allowedTop;
        }

        int allowedFinite() {
            return allowedFinite;
        }

        int vocabularySize() {
            return vocabularySize;
        }
    }
}
