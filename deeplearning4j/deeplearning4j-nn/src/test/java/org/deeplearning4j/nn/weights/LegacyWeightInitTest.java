package org.deeplearning4j.nn.weights;

import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.junit.*;
import org.nd4j.linalg.activations.impl.ActivationCube;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.RandomFactory;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test that {@link WeightInit} is compatible with the corresponding classes which implement {@link IWeightInit}. Mocks
 * Nd4j.randomFactory so that legacy and new implementation can be compared exactly.
 *
 * @author Christian Skarby
 */
public class LegacyWeightInitTest {

    private RandomFactory prevFactory;
    private final static int SEED = 666;

    @Before
    public void setRandomFactory() {
        prevFactory = Nd4j.randomFactory;
        Nd4j.randomFactory = new FixedSeedRandomFactory(prevFactory);
    }

    @After
    public void resetRandomFactory() {
       Nd4j.randomFactory = prevFactory;
    }

    /**
     * Test that param init is identical to legacy implementation
     */
    @Test
    public void initParams() {
        final long[] shape = {5, 5}; // To make identity happy
        final long fanIn = shape[0];
        final long fanOut = shape[1];

        final INDArray expected = Nd4j.create(fanIn * fanOut);
        final INDArray actual = expected.dup();
        for(WeightInit legacyWi: WeightInit.values()) {
            if(legacyWi != WeightInit.DISTRIBUTION) {
                Nd4j.getRandom().setSeed(SEED);
                WeightInitUtil.initWeights(fanIn, fanOut, shape, legacyWi, null, expected);

                Nd4j.getRandom().setSeed(SEED);
                legacyWi.getWeightInitFunction(null).init(fanIn, fanOut, shape, WeightInitUtil.DEFAULT_WEIGHT_INIT_ORDER, actual);

                assertEquals("Incorrect weight initialization for " + legacyWi + "!", expected, actual);
            }
        }
        // TODO: All distributions!

    }

    /**
     * Assumes RandomFactory will only call no-args constructor while this test runs
     */
    public static class FixedSeedRandomFactory extends RandomFactory {
        private final RandomFactory factory;


        public FixedSeedRandomFactory(RandomFactory factory) {
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
