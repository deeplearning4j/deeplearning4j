/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.rng;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.rng.deallocator.NativePack;
import org.nd4j.rng.deallocator.NativeRandomDeallocator;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Basic NativeRandom implementation
 *
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class NativeRandom implements Random {
    protected NativeOps nativeOps;
    protected DataBuffer stateBuffer;
    protected Pointer statePointer;
    protected long seed;
    protected long amplifier;
    protected long generation;
    protected long numberOfElements;
    protected AtomicInteger position = new AtomicInteger(0);
    protected LongPointer hostPointer;
    protected boolean isDestroyed = false;
    protected NativeRandomDeallocator deallocator;

    // special stuff for gaussian
    protected double z0, z1, u0, u1;
    protected boolean generated = false;
    protected double mean = 0.0;
    protected double stdDev = 1.0;

    // hack to attach deallocator
    protected NativePack pack;


    public long getBufferSize() {
        return numberOfElements;
    }

    public int getPosition() {
        return position.get();
    }

    public long getGeneration() {
        return generation;
    }

    public NativeRandom() {
        this(System.currentTimeMillis());
    }

    public NativeRandom(long seed) {
        this(seed, 10000000);
    }

    public NativeRandom(long seed, long numberOfElements) {
        this.amplifier = seed;
        this.generation = 1;
        this.seed = seed;
        this.numberOfElements = numberOfElements;

        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        stateBuffer = Nd4j.getDataBufferFactory().createDouble(numberOfElements);

        init();

        hostPointer = new LongPointer(stateBuffer.addressPointer());

        deallocator = NativeRandomDeallocator.getInstance();

        pack = new NativePack(statePointer.address(), statePointer);

        deallocator.trackStatePointer(pack);
    }

    public abstract void init();

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
            this.amplifier = seed;
            this.position.set(0);
            nativeOps.refreshBuffer(getExtraPointers(), seed, statePointer);
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
        int next = (int) (amplifier == seed ? nextLong() : nextLong() * amplifier + 11);
        return next < 0 ? -1 * next : next;
    }

    @Override
    public int nextInt(int to) {
        int r = nextInt();
        int m = to - 1;
        if ((to & m) == 0) // i.e., bound is a power of 2
            r = (int) ((to * (long) r) >> 31);
        else {
            for (int u = r; u - (r = u % to) + m < 0; u = nextInt());
        }
        return r;
    }

    @Override
    public int nextInt(int a, int n) {
        return nextInt(n - a) + a;
    }

    @Override
    public long nextLong() {
        long next = 0;
        synchronized (this) {
            if (position.get() >= numberOfElements) {
                position.set(0);
                generation++;
            }

            next = hostPointer.get(position.getAndIncrement());

            if (generation > 1)
                next = next ^ generation + 11;

            if (amplifier != seed)
                next = next ^ amplifier + 11;
        }

        return next < 0 ? -1 * next : next;
    }

    public abstract PointerPointer getExtraPointers();

    @Override
    public boolean nextBoolean() {
        return nextInt() % 2 == 0;
    }

    @Override
    public float nextFloat() {
        return (float) nextInt() / (float) Integer.MAX_VALUE;
    }

    @Override
    public double nextDouble() {
        return (double) nextInt() / (double) Integer.MAX_VALUE;
    }

    @Override
    public double nextGaussian() {
        double epsilon = 1e-15;
        double two_pi = 2.0 * 3.14159265358979323846;

        if (!generated) {
            do {
                u0 = nextDouble();
                u1 = nextDouble();
            } while (u0 <= epsilon);

            z0 = Math.sqrt(-2.0 * Math.log(u0)) * Math.cos(two_pi * u1);
            z1 = Math.sqrt(-2.0 * Math.log(u0)) * Math.sin(two_pi * u1);

            generated = true;

            return z0 * stdDev + mean;
        } else {
            generated = false;

            return z1 * stdDev + mean;
        }
    }

    @Override
    public INDArray nextGaussian(int[] shape) {
        return nextGaussian(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextGaussian(long[] shape) {
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
    public INDArray nextGaussian(char order, long[] shape) {
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
    public INDArray nextDouble(long[] shape) {
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
    public INDArray nextDouble(char order, long[] shape) {
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
    public INDArray nextFloat(long[] shape) {
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
    public INDArray nextFloat(char order, long[] shape) {
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
    public INDArray nextInt(long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray nextInt(int n, int[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray nextInt(int n, long[] shape) {
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
    public void reSeed() {
        reSeed(System.currentTimeMillis());
    }

    @Override
    public void reSeed(long amplifier) {
        this.amplifier = amplifier;

        nativeOps.reSeedBuffer(getExtraPointers(), amplifier, getStatePointer());
    }

    @Override
    public void close() throws Exception {
        /*
            Do nothing here, since we use WeakReferences for actual deallocation
         */
    }
}
