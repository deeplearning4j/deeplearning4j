package org.nd4j.linalg.api.rng;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

/**
 * Apache commons based random number generation
 * @author Adam Gibson
 */
public class DefaultRandom implements Random {
    protected RandomGenerator randomGenerator;

    public DefaultRandom(long seed) {
        this.randomGenerator = new MersenneTwister(seed);
    }


    public DefaultRandom(RandomGenerator randomGenerator) {
        this.randomGenerator = randomGenerator;
    }

    @Override
    public void setSeed(int seed) {
       randomGenerator.setSeed(seed);
    }

    @Override
    public void setSeed(int[] seed) {
        randomGenerator.setSeed(seed);
    }

    @Override
    public void setSeed(long seed) {
       randomGenerator.setSeed(seed);
    }

    @Override
    public void nextBytes(byte[] bytes) {
      randomGenerator.nextBytes(bytes);
    }

    @Override
    public int nextInt() {
        return randomGenerator.nextInt();
    }

    @Override
    public int nextInt(int n) {
        return randomGenerator.nextInt(n);
    }

    @Override
    public long nextLong() {
        return randomGenerator.nextLong();
    }

    @Override
    public boolean nextBoolean() {
        return randomGenerator.nextBoolean();
    }

    @Override
    public float nextFloat() {
        return randomGenerator.nextFloat();
    }

    @Override
    public double nextDouble() {
        return randomGenerator.nextDouble();
    }

    @Override
    public double nextGaussian() {
        return randomGenerator.nextGaussian();
    }
}
