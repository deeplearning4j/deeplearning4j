package org.nd4j.rng;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * Basic nativeRandom implementation
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
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public long getSeed() {
        return seed;
    }

    @Override
    public void nextBytes(byte[] bytes) {

    }

    @Override
    public int nextInt() {
        return 0;
    }

    @Override
    public int nextInt(int n) {
        return 0;
    }

    @Override
    public long nextLong() {
        return 0;
    }

    @Override
    public boolean nextBoolean() {
        return false;
    }

    @Override
    public float nextFloat() {
        return 0;
    }

    @Override
    public double nextDouble() {
        return 0;
    }

    @Override
    public double nextGaussian() {
        return 0;
    }

    @Override
    public INDArray nextGaussian(int[] shape) {
        return null;
    }

    @Override
    public INDArray nextGaussian(char order, int[] shape) {
        return null;
    }

    @Override
    public INDArray nextDouble(int[] shape) {
        return null;
    }

    @Override
    public INDArray nextDouble(char order, int[] shape) {
        return null;
    }

    @Override
    public INDArray nextFloat(int[] shape) {
        return null;
    }

    @Override
    public INDArray nextInt(int[] shape) {
        return null;
    }

    @Override
    public INDArray nextInt(int n, int[] shape) {
        return null;
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
}
