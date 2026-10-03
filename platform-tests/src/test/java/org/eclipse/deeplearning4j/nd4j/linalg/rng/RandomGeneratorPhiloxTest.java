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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.random.custom.DistributionUniform;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The native RandomGenerator draws Philox4x32-10 blocks: the key is the root state and the counter
 * is (index, node state). Its earlier index hash read only the low 32 bits of each state and
 * correlated an element's consecutive sampler draws (-0.05), which biased Poisson and gamma
 * samples. Uniform fills now equal {@link PhiloxReference} bit for bit on every backend, whether
 * a seed argument or {@code Nd4j.getRandom()} sets the states.
 */
@Tag(TagNames.RNG)
@NativeTag
public class RandomGeneratorPhiloxTest extends BaseNd4jTestWithBackends {

    /** Random123's known-answer vectors for Philox4x32-10 hold for the reference. */
    @Test
    public void referenceMatchesKnownAnswers() {
        assertArrayEquals(new int[]{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8},
                PhiloxReference.block(0, 0, 0, 0, 0, 0));
        assertArrayEquals(new int[]{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd},
                PhiloxReference.block(-1, -1, -1, -1, -1, -1));
        assertArrayEquals(new int[]{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1},
                PhiloxReference.block(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0));
    }

    /** A seeded randomuniform fill is the reference's draw at each index, in FLOAT and DOUBLE. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void seededUniformIsPhilox(Nd4jBackend backend) {
        int length = 4099;
        for (long seed : new long[]{42L, 119L, 1L + (1L << 40), -7L}) {
            long root = PhiloxReference.seededRoot(seed);
            long node = PhiloxReference.seededNode(seed);
            float[] floats = seededUniform(DataType.FLOAT, length, seed).data().asFloat();
            double[] doubles = seededUniform(DataType.DOUBLE, length, seed).data().asDouble();
            for (int i = 0; i < length; i++) {
                assertEquals(Float.floatToIntBits(PhiloxReference.uniformFloat(root, node, i)),
                        Float.floatToIntBits(floats[i]), "seed " + seed + ", FLOAT index " + i);
                assertEquals(Double.doubleToLongBits(PhiloxReference.uniformDouble(root, node, i)),
                        Double.doubleToLongBits(doubles[i]), "seed " + seed + ", DOUBLE index " + i);
            }
        }
    }

    /** An unseeded uniform op draws at the states Nd4j.getRandom() holds, high bits included. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void threadGeneratorStatesReachTheDraws(Nd4jBackend backend) {
        long root = 0x123456789abcdef0L;
        long node = 0x0fedcba987654321L;
        Nd4j.getRandom().setStates(root, node);
        int length = 257;
        INDArray out = Nd4j.create(DataType.FLOAT, length);
        Nd4j.exec(new DistributionUniform(Nd4j.createFromArray((long) length), out, 0.0, 1.0, DataType.FLOAT));
        float[] values = out.data().asFloat();
        for (int i = 0; i < length; i++) {
            assertEquals(Float.floatToIntBits(PhiloxReference.uniformFloat(root, node, i)),
                    Float.floatToIntBits(values[i]), "index " + i);
        }
    }

    /**
     * Nd4j.getRandom().setSeed(s) sets the states a seed argument s sets, so an unseeded op after it
     * draws what the seeded op draws. CPU and CUDA once sign-extended 0xdeadbeef into the node state.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void setSeedMatchesSeedArgument(Nd4jBackend backend) {
        long seed = 0x5eed0003L;
        Nd4j.getRandom().setSeed(seed);
        assertEquals(PhiloxReference.seededRoot(seed), Nd4j.getRandom().rootState());
        assertEquals(PhiloxReference.seededNode(seed), Nd4j.getRandom().nodeState());
        float[] unseeded = seededUniform(DataType.FLOAT, 1000, 0L).data().asFloat();
        float[] seeded = seededUniform(DataType.FLOAT, 1000, seed).data().asFloat();
        assertArrayEquals(seeded, unseeded);
    }

    /** Seeds that differ only above bit 31 draw different streams; the old hash ignored those bits. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void highSeedBitsChangeTheStream(Nd4jBackend backend) {
        float[] low = seededUniform(DataType.FLOAT, 64, 5L).data().asFloat();
        float[] high = seededUniform(DataType.FLOAT, 64, 5L + (1L << 40)).data().asFloat();
        int equal = 0;
        for (int i = 0; i < low.length; i++) {
            if (low[i] == high[i]) equal++;
        }
        assertTrue(equal < 4, equal + " of 64 draws are equal");
    }

    /** Neighbouring draws of a stream are uncorrelated and uniform: within 5 standard errors. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void neighbouringDrawsAreUncorrelated(Nd4jBackend backend) {
        int n = 1 << 20;
        float[] u = seededUniform(DataType.FLOAT, n, 123456789L).data().asFloat();
        double mean = 0;
        for (float v : u) mean += v;
        mean /= n;
        assertTrue(Math.abs(mean - 0.5) < 5 * Math.sqrt(1.0 / 12.0 / n), "mean " + mean);
        double bound = 5 / Math.sqrt(n);
        for (int lag = 1; lag <= 3; lag++) {
            double r = lagCorrelation(u, mean, lag);
            assertTrue(Math.abs(r) < bound, "lag-" + lag + " correlation " + r + " (bound " + bound + ")");
        }
    }

    private static INDArray seededUniform(DataType dataType, int length, long seed) {
        INDArray out = Nd4j.create(dataType, length);
        Nd4j.exec(DynamicCustomOp.builder("randomuniform")
                .addInputs(Nd4j.createFromArray((long) length))
                .addOutputs(out)
                .addFloatingPointArguments(0.0, 1.0)
                .addIntegerArguments(dataType.toInt(), seed)
                .build());
        return out;
    }

    private static double lagCorrelation(float[] u, double mean, int lag) {
        double covariance = 0;
        double variance = 0;
        for (int i = 0; i < u.length; i++) {
            double d = u[i] - mean;
            variance += d * d;
            if (i + lag < u.length) covariance += d * (u[i + lag] - mean);
        }
        return covariance / variance;
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
