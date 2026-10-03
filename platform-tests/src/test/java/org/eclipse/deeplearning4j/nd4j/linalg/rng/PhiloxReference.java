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
package org.eclipse.deeplearning4j.nd4j.linalg.rng;

/**
 * Philox4x32-10 as the native RandomGenerator computes it (libnd4j/include/graph/RandomGenerator.h):
 * the key is the root state and the counter is (index, node state). Tests compare native draws
 * with it bit for bit.
 */
public final class PhiloxReference {

    private PhiloxReference() {
    }

    /** The four 32-bit words of Philox4x32-10 for a counter and a key. */
    public static int[] block(int c0, int c1, int c2, int c3, int k0, int k1) {
        int[] c = {c0, c1, c2, c3};
        for (int pass = 0; pass < 10; pass++) {
            if (pass > 0) {
                k0 += 0x9E3779B9;
                k1 += 0xBB67AE85;
            }
            long product0 = 0xD2511F53L * (c[0] & 0xFFFFFFFFL);
            long product1 = 0xCD9E8D57L * (c[2] & 0xFFFFFFFFL);
            int next0 = (int) (product1 >>> 32) ^ c[1] ^ k0;
            int next2 = (int) (product0 >>> 32) ^ c[3] ^ k1;
            c[1] = (int) product1;
            c[3] = (int) product0;
            c[0] = next0;
            c[2] = next2;
        }
        return c;
    }

    /** The block of a generator with these states at index. */
    public static int[] block(long root, long node, long index) {
        return block((int) index, (int) (index >>> 32), (int) node, (int) (node >>> 32), (int) root, (int) (root >>> 32));
    }

    /** RandomGenerator::relativeT&lt;float&gt;: [0, 1) from the top 23 bits of word 0. */
    public static float uniformFloat(long root, long node, long index) {
        return Float.intBitsToFloat(0x3f800000 | (block(root, node, index)[0] >>> 9)) - 1.0f;
    }

    /** RandomGenerator::relativeT&lt;double&gt;: [0, 1) from the top 52 bits of words 1 and 0. */
    public static double uniformDouble(long root, long node, long index) {
        int[] words = block(root, node, index);
        long bits = ((long) words[1] << 32) | (words[0] & 0xFFFFFFFFL);
        return Double.longBitsToDouble((0x3FFL << 52) | (bits >>> 12)) - 1.0;
    }

    /** The root state a random op's seed argument sets (helpers::applySeedArgument). */
    public static long seededRoot(long seed) {
        return seed;
    }

    /** The node state a random op's seed argument sets (helpers::applySeedArgument). */
    public static long seededNode(long seed) {
        return seed ^ 0xdeadbeefL;
    }
}
