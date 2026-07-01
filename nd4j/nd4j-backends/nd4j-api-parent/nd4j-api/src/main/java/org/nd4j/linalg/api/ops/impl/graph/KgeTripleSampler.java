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
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.graph;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Negative sampling for knowledge-graph embedding (KGE) training.
 *
 * <p>KGE models are trained to score the observed triples {@code (head, relation, tail)} higher
 * than corrupted ones. This utility generates corrupted triples by replacing the head or tail with
 * a random entity, optionally <em>filtered</em> so a "negative" is never an actually-true triple.
 * Pure CPU/JVM logic, reproducible from {@code seed}.
 */
public final class KgeTripleSampler {

    private KgeTripleSampler() { }

    /** Max rejection-sampling attempts to find a filtered negative before failing loudly. */
    private static final int MAX_CORRUPTION_ATTEMPTS = 100;

    private static long encode(int h, int r, int t, int numEntities, int numRelations) {
        return ((long) h * numRelations + r) * (long) numEntities + t;
    }

    /** Build the set of known true triples — pass to {@link #corrupt} for filtered sampling. */
    public static Set<Long> knownSet(int[][] triples, int numEntities, int numRelations) {
        Set<Long> s = new HashSet<>();
        for (int[] tr : triples) {
            s.add(encode(tr[0], tr[1], tr[2], numEntities, numRelations));
        }
        return s;
    }

    /**
     * Generate {@code negPerPos} corrupted triples for each positive, by replacing the head or tail
     * (50/50) with a uniformly random entity; corruptions that happen to be true triples (present in
     * {@code known}) are rejected and re-drawn.
     *
     * @param positives    positive triples, each {@code {head, relation, tail}}
     * @param numEntities  entity vocabulary size
     * @param numRelations relation vocabulary size
     * @param negPerPos    negatives to generate per positive
     * @param known        known true triples (from {@link #knownSet}); pass {@code null} for unfiltered
     * @param seed         RNG seed
     * @return negative triples, length {@code positives.length * negPerPos}
     */
    public static int[][] corrupt(int[][] positives, int numEntities, int numRelations,
                                  int negPerPos, Set<Long> known, long seed) {
        Random rng = new Random(seed);
        int[][] negs = new int[positives.length * negPerPos][3];
        int idx = 0;
        for (int[] pos : positives) {
            for (int k = 0; k < negPerPos; k++) {
                int h, t;
                int r = pos[1];
                int tries = 0;
                do {
                    h = pos[0];
                    t = pos[2];
                    if (rng.nextBoolean()) {
                        h = rng.nextInt(numEntities);     // corrupt head
                    } else {
                        t = rng.nextInt(numEntities);     // corrupt tail
                    }
                    tries++;
                } while (known != null && known.contains(encode(h, r, t, numEntities, numRelations))
                        && tries < MAX_CORRUPTION_ATTEMPTS);
                // Filtered sampling must never emit a known-true triple as a "negative". If we
                // exhausted the attempt budget, the entity space is too small/saturated to corrupt
                // this positive — fail loudly rather than poison the batch with a true triple.
                if (known != null && known.contains(encode(h, r, t, numEntities, numRelations))) {
                    throw new IllegalStateException(
                            "Could not sample a filtered negative for triple {" + pos[0] + ", " + r + ", "
                            + pos[2] + "} after " + MAX_CORRUPTION_ATTEMPTS + " attempts; the entity space"
                            + " (numEntities=" + numEntities + ") is too small or too saturated for filtered"
                            + " negative sampling. Increase numEntities, reduce negPerPos, or pass known=null"
                            + " for unfiltered sampling.");
                }
                negs[idx][0] = h;
                negs[idx][1] = r;
                negs[idx][2] = t;
                idx++;
            }
        }
        return negs;
    }

    /** Extract one column (0=head, 1=relation, 2=tail) of a triple array as an int[] for gathering. */
    public static int[] column(int[][] triples, int col) {
        int[] out = new int[triples.length];
        for (int i = 0; i < triples.length; i++) {
            out[i] = triples[i][col];
        }
        return out;
    }
}
