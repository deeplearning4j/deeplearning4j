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
 *  * distributed under the LICENSE is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.graph;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Random;

/**
 * Deterministic train / validation / test edge splitter.
 *
 * <p>Given an edge list of shape {@code [E, 2]} (each row is an (src, dst) node-index pair),
 * the splitter shuffles the edges with a fixed seed and partitions them into three
 * <em>disjoint</em> subsets with sizes proportional to the requested fractions.
 * The union of the three subsets equals the full input edge list.
 *
 * <p>Usage:
 * <pre>{@code
 *   EdgeSplitter splitter = new EdgeSplitter(42L);
 *   EdgeSplitter.Split split = splitter.split(edgeList, 0.8, 0.1);
 *   // split.getTrain() has ~80% of edges, split.getVal() ~10%, split.getTest() ~10%
 * }</pre>
 *
 * <p>The same seed always produces the same split from the same input (deterministic).
 */
public class EdgeSplitter {

    private final long seed;

    /**
     * @param seed random seed for reproducibility
     */
    public EdgeSplitter(long seed) {
        this.seed = seed;
    }

    // -------------------------------------------------------------------------
    // Split result holder
    // -------------------------------------------------------------------------

    /**
     * Holds the three disjoint edge-list arrays resulting from a split.
     * Each array has shape {@code [K, 2]} for varying K.
     */
    public static final class Split {
        private final INDArray train;
        private final INDArray val;
        private final INDArray test;

        private Split(INDArray train, INDArray val, INDArray test) {
            this.train = train;
            this.val   = val;
            this.test  = test;
        }

        /** Edge list for the training set, shape {@code [trainSize, 2]}. */
        public INDArray getTrain() { return train; }

        /** Edge list for the validation set, shape {@code [valSize, 2]}. */
        public INDArray getVal()   { return val; }

        /** Edge list for the test set, shape {@code [testSize, 2]}. */
        public INDArray getTest()  { return test; }
    }

    // -------------------------------------------------------------------------
    // Main split method
    // -------------------------------------------------------------------------

    /**
     * Split an edge list into train / val / test subsets.
     *
     * @param edgeList  INDArray of shape {@code [E, 2]} with integer (or float-cast) node indices.
     * @param trainFrac fraction of edges in the training set; must be in (0, 1).
     * @param valFrac   fraction of edges in the validation set; must be in [0, 1).
     *                  The test fraction is {@code 1 - trainFrac - valFrac}.
     * @return a {@link Split} with three disjoint edge arrays whose union equals {@code edgeList}.
     * @throws IllegalArgumentException if fractions are invalid or the edge list is not [E,2].
     */
    public Split split(INDArray edgeList, double trainFrac, double valFrac) {
        if (edgeList.rank() != 2 || edgeList.size(1) != 2) {
            throw new IllegalArgumentException(
                    "edgeList must have shape [E, 2], got " + Arrays.toString(edgeList.shape()));
        }
        if (trainFrac <= 0 || trainFrac >= 1) {
            throw new IllegalArgumentException("trainFrac must be in (0,1), got " + trainFrac);
        }
        if (valFrac < 0 || valFrac >= 1) {
            throw new IllegalArgumentException("valFrac must be in [0,1), got " + valFrac);
        }
        if (trainFrac + valFrac >= 1.0 + 1e-9) {
            throw new IllegalArgumentException(
                    "trainFrac + valFrac must be < 1, got " + (trainFrac + valFrac));
        }

        int E = (int) edgeList.size(0);
        if (E == 0) throw new IllegalArgumentException("edgeList is empty (E=0)");

        // Create a shuffled index array.
        int[] perm = new int[E];
        for (int i = 0; i < E; i++) perm[i] = i;
        shuffle(perm, new Random(seed));

        // Determine split boundaries.
        int trainEnd = (int) Math.round(trainFrac * E);
        int valEnd   = trainEnd + (int) Math.round(valFrac * E);
        // Test gets everything left over; clamp to E.
        valEnd  = Math.min(valEnd, E);
        trainEnd = Math.min(trainEnd, E);

        // Guard: ensure train gets at least 1 and test gets at least 0.
        // (With very small E these could collide; we just let it happen — callers
        //  should ensure E is large enough for the requested fractions.)

        INDArray train = gatherRows(edgeList, perm, 0,        trainEnd);
        INDArray val   = gatherRows(edgeList, perm, trainEnd, valEnd);
        INDArray test  = gatherRows(edgeList, perm, valEnd,   E);

        return new Split(train, val, test);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /** Fisher-Yates shuffle on an int[] array. */
    private static void shuffle(int[] arr, Random rng) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        }
    }

    /**
     * Gather rows {@code perm[from]..perm[to-1]} from {@code src} (shape [E,2])
     * into a new INDArray of shape {@code [to-from, 2]}.
     */
    private static INDArray gatherRows(INDArray src, int[] perm, int from, int to) {
        int count = to - from;
        if (count == 0) {
            // Return an empty [0,2] array.
            return Nd4j.create(DataType.LONG, 0, 2);
        }
        long[] data = new long[count * 2];
        for (int i = 0; i < count; i++) {
            int row = perm[from + i];
            data[i * 2]     = src.getLong(row, 0);
            data[i * 2 + 1] = src.getLong(row, 1);
        }
        return Nd4j.createFromArray(data).reshape(count, 2);
    }
}
