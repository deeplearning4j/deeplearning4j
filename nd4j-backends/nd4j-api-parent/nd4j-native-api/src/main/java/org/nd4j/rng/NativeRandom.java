package org.nd4j.rng;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Basic NativeRandom implementation
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class NativeRandom implements Random {
    protected NativeOps nativeOps;
    protected DataBuffer stateBuffer;
    protected Pointer statePointer;
    protected long seed;
    protected long numberOfElements;
    protected AtomicInteger position = new AtomicInteger(0);
    protected LongPointer hostPointer;
    protected boolean isDestroyed = false;

    public NativeRandom() {
        this(System.currentTimeMillis());
    }

    public NativeRandom(long seed) {
        this(seed, 5000000);
    }

    public NativeRandom(long seed, long numberOfElements) {
        this.numberOfElements = numberOfElements;

        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        stateBuffer = Nd4j.getDataBufferFactory().createDouble(numberOfElements);

        statePointer = nativeOps.initRandom(seed, numberOfElements, stateBuffer.addressPointer());

        hostPointer = new LongPointer(stateBuffer.addressPointer());
    }

    @Override
    public void setSeed(int seed) {
        setSeed((long) seed);
    }

    @Override
    public void setSeed(int[] seed) {
        long sd = 0;
        for (int em : seed) {
            sd *= em;
        }
        setSeed(sd);
    }

    @Override
    public void setSeed(long seed) {
        synchronized (this) {
            this.seed = seed;
            nativeOps.refreshBuffer(seed, statePointer);
        }
    }

    @Override
    public long getSeed() {
        return seed;
    }

    @Override
    public void nextBytes(byte[] bytes) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int nextInt() {
        int next = (int) nextLong();
        return next < 0 ? -1 * next : next;
    }

    @Override
    public int nextInt(int to) {
        int r = nextInt();
        int m = to - 1;
        if ((to & m) == 0)  // i.e., bound is a power of 2
            r = (int) ((to * (long) r) >> 31);
        else {
            for (int u = r;
                 u - (r = u % to) + m < 0;
                 u = nextInt());
        }
        return r;
    }

    @Override
    public long nextLong() {
        long next = 0;
        synchronized (this) {
            if (position.get() >= numberOfElements) {
                position.set(0);
            }

            next = hostPointer.get(position.getAndIncrement());
        }

        return next < 0 ? -1 * next : next;
    }

    @Override
    public boolean nextBoolean() {
        return nextInt() % 2 == 0;
    }

    @Override
    public float nextFloat() {
        return  (float) nextInt() / (float)  Integer.MAX_VALUE;
    }

    @Override
    public double nextDouble() {
        return (double) nextInt() / (double)  Integer.MAX_VALUE;
    }

    @Override
    public double nextGaussian() {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray nextGaussian(int[] shape) {
        return nextGaussian(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextGaussian(char order, int[] shape) {
        INDArray array = Nd4j.createUninitialized(shape, order);
        GaussianDistribution op = new GaussianDistribution(array, 0.0, 1.0);
        Nd4j.getExecutioner().exec(op, this);

        return array;
    }

    @Override
    public INDArray nextDouble(int[] shape) {
        return nextDouble(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextDouble(char order, int[] shape) {
        INDArray array = Nd4j.createUninitialized(shape, order);
        UniformDistribution op = new UniformDistribution(array, 0.0, 1.0);
        Nd4j.getExecutioner().exec(op, this);

        return array;
    }

    @Override
    public INDArray nextFloat(int[] shape) {
        return nextFloat(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextFloat(char order, int[] shape) {
        INDArray array = Nd4j.createUninitialized(shape, order);
        UniformDistribution op = new UniformDistribution(array, 0.0, 1.0);
        Nd4j.getExecutioner().exec(op, this);

        return array;
    }

    @Override
    public INDArray nextInt(int[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray nextInt(int n, int[] shape) {
        throw new UnsupportedOperationException();
    }

    /**
     * This method returns pointer to RNG state structure.
     * Please note: DefaultRandom implementation returns NULL here, making it impossible to use with RandomOps
     *
     * @return
     */
    @Override
    public Pointer getStatePointer() {
        return statePointer;
    }

    /**
     * This method returns pointer to RNG buffer
     *
     * @return
     */
    @Override
    public DataBuffer getStateBuffer() {
        return stateBuffer;
    }

    @Override
    public void close() throws Exception {
        /*
            Do nothing here, since we use WeakReferences for actual deallocation
         */
    }
}
