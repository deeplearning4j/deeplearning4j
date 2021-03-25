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
package org.deeplearning4j.nn.weights;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.distribution.*;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.RandomFactory;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Legacy Weight Init Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class LegacyWeightInitTest extends BaseDL4JTest {

    private RandomFactory prevFactory;

    private final static int SEED = 666;

    private final static List<Distribution> distributions = Arrays.asList(new LogNormalDistribution(12.3, 4.56), new BinomialDistribution(3, 0.3), new NormalDistribution(0.666, 0.333), new UniformDistribution(-1.23, 4.56), new OrthogonalDistribution(3.45), new TruncatedNormalDistribution(0.456, 0.123), new ConstantDistribution(666));

    @BeforeEach
    void setRandomFactory() {
        prevFactory = Nd4j.randomFactory;
        Nd4j.randomFactory = new FixedSeedRandomFactory(prevFactory);
    }

    @AfterEach
    void resetRandomFactory() {
        Nd4j.randomFactory = prevFactory;
    }

    /**
     * Test that param init is identical to legacy implementation
     */
    @Test
    @DisplayName("Init Params")
    void initParams() {
        // To make identity happy
        final long[] shape = { 5, 5 };
        final long fanIn = shape[0];
        final long fanOut = shape[1];
        final INDArray inLegacy = Nd4j.create(DataType.DOUBLE,fanIn * fanOut);
        final INDArray inTest = inLegacy.dup();
        for (WeightInit legacyWi : WeightInit.values()) {
            if (legacyWi != WeightInit.DISTRIBUTION) {
                Nd4j.getRandom().setSeed(SEED);
                final INDArray expected = WeightInitUtil.
                        initWeights(fanIn, fanOut, shape, legacyWi, null, inLegacy)
                        .castTo(DataType.DOUBLE);
                Nd4j.getRandom().setSeed(SEED);
                final INDArray actual = legacyWi.getWeightInitFunction()
                        .init(fanIn, fanOut, shape,
                                WeightInitUtil.DEFAULT_WEIGHT_INIT_ORDER, inTest)
                        .castTo(DataType.DOUBLE);
                assertArrayEquals(shape, actual.shape(),"Incorrect shape for " + legacyWi + "!");
                assertEquals( expected, actual,"Incorrect weight initialization for " + legacyWi + "!");
            }
        }
    }

    /**
     * Test that param init is identical to legacy implementation
     */
    @Test
    @DisplayName("Init Params From Distribution")
    @Execution(ExecutionMode.SAME_THREAD)
    @Disabled(TagNames.NEEDS_VERIFY)
    void initParamsFromDistribution() {
        // To make identity happy
        final long[] shape = { 3, 7 };
        final long fanIn = shape[0];
        final long fanOut = shape[1];
        final INDArray inLegacy = Nd4j.create(DataType.DOUBLE,fanIn * fanOut);
        final INDArray inTest = inLegacy.dup();
        for (Distribution dist : distributions) {
            Nd4j.getRandom().setSeed(SEED);
            final INDArray expected = WeightInitUtil
                    .initWeights(fanIn, fanOut, shape, WeightInit.DISTRIBUTION,
                            Distributions.createDistribution(dist), inLegacy)
                    .castTo(DataType.DOUBLE);
            final INDArray actual = new WeightInitDistribution(dist)
                    .init(fanIn, fanOut, shape, WeightInitUtil.DEFAULT_WEIGHT_INIT_ORDER,
                            inTest).castTo(DataType.DOUBLE);
            assertArrayEquals(shape, actual.shape(),"Incorrect shape for " + dist.getClass().getSimpleName() + "!");
            assertEquals( expected, actual,"Incorrect weight initialization for " + dist.getClass().getSimpleName() + "!");
        }
    }

    /**
     * Test that weight inits can be serialized and de-serialized in JSON format
     */
    @Test
    @DisplayName("Serialize Deserialize Json")
    void serializeDeserializeJson() throws IOException {
        // To make identity happy
        final long[] shape = { 5, 5 };
        final long fanIn = shape[0];
        final long fanOut = shape[1];
        final ObjectMapper mapper = JsonMappers.getMapper();
        final INDArray inBefore = Nd4j.create(fanIn * fanOut);
        final INDArray inAfter = inBefore.dup();
        // Just use to enum to loop over all strategies
        for (WeightInit legacyWi : WeightInit.values()) {
            if (legacyWi != WeightInit.DISTRIBUTION) {
                Nd4j.getRandom().setSeed(SEED);
                final IWeightInit before = legacyWi.getWeightInitFunction();
                final INDArray expected = before.init(fanIn, fanOut, shape, inBefore.ordering(), inBefore);
                final String json = mapper.writeValueAsString(before);
                final IWeightInit after = mapper.readValue(json, IWeightInit.class);
                Nd4j.getRandom().setSeed(SEED);
                final INDArray actual = after.init(fanIn, fanOut, shape, inAfter.ordering(), inAfter);
                assertArrayEquals( shape, actual.shape(),"Incorrect shape for " + legacyWi + "!");
                assertEquals(expected, actual,"Incorrect weight initialization for " + legacyWi + "!");
            }
        }
    }

    /**
     * Test that distribution can be serialized and de-serialized in JSON format
     */
    @Test
    @DisplayName("Serialize Deserialize Distribution Json")
    void serializeDeserializeDistributionJson() throws IOException {
        // To make identity happy
        final long[] shape = { 3, 7 };
        final long fanIn = shape[0];
        final long fanOut = shape[1];
        final ObjectMapper mapper = JsonMappers.getMapper();
        final INDArray inBefore = Nd4j.create(fanIn * fanOut);
        final INDArray inAfter = inBefore.dup();
        for (Distribution dist : distributions) {
            Nd4j.getRandom().setSeed(SEED);
            final IWeightInit before = new WeightInitDistribution(dist);
            final INDArray expected = before.init(fanIn, fanOut, shape, inBefore.ordering(), inBefore);
            final String json = mapper.writeValueAsString(before);
            final IWeightInit after = mapper.readValue(json, IWeightInit.class);
            Nd4j.getRandom().setSeed(SEED);
            final INDArray actual = after.init(fanIn, fanOut, shape, inAfter.ordering(), inAfter);
            assertArrayEquals(shape, actual.shape(),"Incorrect shape for " + dist.getClass().getSimpleName() + "!");
            assertEquals(expected, actual,"Incorrect weight initialization for " + dist.getClass().getSimpleName() + "!");
        }
    }

    /**
     * Test equals and hashcode implementation. Redundant as one can trust Lombok on this??
     */
    @Test
    @DisplayName("Equals And Hash Code")
    void equalsAndHashCode() {
        WeightInit lastInit = WeightInit.values()[WeightInit.values().length - 1];
        for (WeightInit legacyWi : WeightInit.values()) {
            if (legacyWi != WeightInit.DISTRIBUTION) {
                assertEquals(legacyWi.getWeightInitFunction(), legacyWi.getWeightInitFunction(), "Shall be equal!");
                assertNotEquals(lastInit.getWeightInitFunction(), legacyWi.getWeightInitFunction(), "Shall not be equal!");
                if (legacyWi != WeightInit.NORMAL && legacyWi != WeightInit.LECUN_NORMAL) {
                    lastInit = legacyWi;
                }
            }
        }
        Distribution lastDist = distributions.get(distributions.size() - 1);
        for (Distribution distribution : distributions) {
            assertEquals(new WeightInitDistribution(distribution), new WeightInitDistribution(distribution.clone()), "Shall be equal!");
            assertNotEquals(new WeightInitDistribution(lastDist), new WeightInitDistribution(distribution), "Shall not be equal!");
            lastDist = distribution;
        }
    }

    /**
     * Assumes RandomFactory will only call no-args constructor while this test runs
     */
    @DisplayName("Fixed Seed Random Factory")
    private static class FixedSeedRandomFactory extends RandomFactory {

        private final RandomFactory factory;

        private FixedSeedRandomFactory(RandomFactory factory) {
            super(factory.getRandom().getClass());
            this.factory = factory;
        }

        @Override
        public Random getRandom() {
            return getNewRandomInstance(SEED);
        }

        @Override
        public Random getNewRandomInstance() {
            return factory.getNewRandomInstance();
        }

        @Override
        public Random getNewRandomInstance(long seed) {
            return factory.getNewRandomInstance(seed);
        }

        @Override
        public Random getNewRandomInstance(long seed, long size) {
            return factory.getNewRandomInstance(seed, size);
        }
    }
}
