package org.nd4j.linalg.api.rng;

import it.unimi.dsi.util.XoRoShiRo128PlusRandom;
import org.apache.commons.math3.random.RandomGenerator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * A faster random implementation using <a href="http://dsiutils.di.unimi.it/docs/it/unimi/dsi/util/XoRoShiRo128PlusRandom.html">XoRoShiRo128PlusRandom</a>
 * The content of this class is the same as DefaultRandom, but using XoRoShiRo128PlusRandom for generating
 * random numbers. This yields substantial performance improvements for dropConnect, for instance.
 * Created by Fabien Campagne on 8/8/16.
 */

public class FasterRandom implements Random, RandomGenerator {
    protected XoRoShiRo128PlusRandom randomGenerator;

    protected long seed;

    /**
     * Initialize with a System.currentTimeMillis()
     * seed
     */
    public FasterRandom() {
        this(System.currentTimeMillis());
    }

    public FasterRandom(long seed) {
        this.seed = seed;
        this.randomGenerator = new XoRoShiRo128PlusRandom(seed);
    }

    public FasterRandom(XoRoShiRo128PlusRandom randomGenerator) {
        this.randomGenerator = randomGenerator;
    }

    @Override
    public void setSeed(int seed) {
        this.seed = (long) seed;
        getRandomGenerator().setSeed(seed);
    }


    @Override
    public void setSeed(int[] seed) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setSeed(long seed) {
        this.seed = seed;
        getRandomGenerator().setSeed(seed);
    }

    @Override
    public void nextBytes(byte[] bytes) {
        getRandomGenerator().nextBytes(bytes);
    }

    @Override
    public int nextInt() {
        return getRandomGenerator().nextInt();
    }

    @Override
    public int nextInt(int n) {
        return getRandomGenerator().nextInt(n);
    }

    @Override
    public long nextLong() {
        return getRandomGenerator().nextLong();
    }

    @Override
    public boolean nextBoolean() {
        return getRandomGenerator().nextBoolean();
    }

    @Override
    public float nextFloat() {
        return getRandomGenerator().nextFloat();
    }

    @Override
    public double nextDouble() {
        return getRandomGenerator().nextDouble();
    }

    @Override
    public double nextGaussian() {
        return getRandomGenerator().nextGaussian();
    }

    @Override
    public INDArray nextGaussian(int[] shape) {
        return nextGaussian(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextGaussian(char order, int[] shape) {
        int length = ArrayUtil.prod(shape);
        INDArray ret = Nd4j.create(shape, order);

        DataBuffer data = ret.data();
        for (int i = 0; i < length; i++) {
            data.put(i, nextGaussian());
        }

        return ret;
    }

    @Override
    public INDArray nextDouble(int[] shape) {
        return nextDouble(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextDouble(char order, int[] shape) {
        int length = ArrayUtil.prod(shape);
        INDArray ret = Nd4j.create(shape, order);

        DataBuffer data = ret.data();
        for (int i = 0; i < length; i++) {
            data.put(i, nextDouble());
        }

        return ret;
    }

    @Override
    public INDArray nextFloat(int[] shape) {
        int length = ArrayUtil.prod(shape);
        INDArray ret = Nd4j.create(shape);

        DataBuffer data = ret.data();
        for (int i = 0; i < length; i++) {
            data.put(i, nextFloat());
        }

        return ret;
    }

    @Override
    public INDArray nextInt(int[] shape) {
        int length = ArrayUtil.prod(shape);
        INDArray ret = Nd4j.create(shape);

        DataBuffer data = ret.data();
        for (int i = 0; i < length; i++) {
            data.put(i, nextInt());
        }

        return ret;
    }

    @Override
    public INDArray nextInt(int n, int[] shape) {
        int length = ArrayUtil.prod(shape);
        INDArray ret = Nd4j.create(shape);

        DataBuffer data = ret.data();
        for (int i = 0; i < length; i++) {
            data.put(i, nextInt(n));
        }

        return ret;
    }


    public synchronized XoRoShiRo128PlusRandom getRandomGenerator() {
        return randomGenerator;
    }

    public synchronized long getSeed() {
        return this.seed;
    }

}
