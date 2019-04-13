package org.deeplearning4j.arbiter.optimize.genetic;

import org.apache.commons.math3.random.RandomGenerator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class TestRandomGenerator implements RandomGenerator {
    private final int[] intRandomNumbers;
    private int currentIntIdx = 0;
    private final double[] doubleRandomNumbers;
    private int currentDoubleIdx = 0;


    public TestRandomGenerator(int[] intRandomNumbers, double[] doubleRandomNumbers) {
        this.intRandomNumbers = intRandomNumbers;
        this.doubleRandomNumbers = doubleRandomNumbers;
    }

    @Override
    public void setSeed(int i) {

    }

    @Override
    public void setSeed(int[] ints) {

    }

    @Override
    public void setSeed(long l) {

    }

    @Override
    public void nextBytes(byte[] bytes) {

    }

    @Override
    public int nextInt() {
        return intRandomNumbers[currentIntIdx++];
    }

    @Override
    public int nextInt(int i) {
        return intRandomNumbers[currentIntIdx++];
    }

    @Override
    public long nextLong() {
        throw new NotImplementedException();
    }

    @Override
    public boolean nextBoolean() {
        throw new NotImplementedException();
    }

    @Override
    public float nextFloat() {
        throw new NotImplementedException();
    }

    @Override
    public double nextDouble() {
        return doubleRandomNumbers[currentDoubleIdx++];
    }

    @Override
    public double nextGaussian() {
        throw new NotImplementedException();
    }
}
