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

package org.nd4j.linalg.api.rng;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Apache commons based random number generation
 *
 * Please note: this implementation can't be used for NativeOps execution
 *
 * @author Adam Gibson
 */
// TODO: make this op compatible with NativeOpExecutioner
public class DefaultRandom implements Random, RandomGenerator {
    protected RandomGenerator randomGenerator;
    protected long seed;

    /**
     * Initialize with a System.currentTimeMillis()
     * seed
     */
    public DefaultRandom() {
        this(System.currentTimeMillis());
    }

    public DefaultRandom(long seed) {
        this.seed = seed;
        this.randomGenerator = new SynchronizedRandomGenerator(new MersenneTwister(seed));
    }

    public DefaultRandom(RandomGenerator randomGenerator) {
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
    public INDArray nextGaussian(long[] shape) {
        return nextGaussian(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextGaussian(int[] shape) {
        return nextGaussian(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextGaussian(char order, int[] shape) {
        return nextGaussian(order, ArrayUtil.toLongArray(shape));
    }

    @Override
    public INDArray nextGaussian(char order, long[] shape) {
        long length = ArrayUtil.prodLong(shape);
        INDArray ret = Nd4j.create(shape, order);

        DataBuffer data = ret.data();
        for (long i = 0; i < length; i++) {
            data.put(i, nextGaussian());
        }

        return ret;
    }

    @Override
    public INDArray nextDouble(long[] shape) {
        return nextDouble(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextDouble(int[] shape) {
        return nextDouble(Nd4j.order(), shape);
    }


    @Override
    public INDArray nextDouble(char order, int[] shape) {
        return nextDouble(order, ArrayUtil.toLongArray(shape));
    }

    @Override
    public INDArray nextDouble(char order, long[] shape) {
        long length = ArrayUtil.prodLong(shape);
        INDArray ret = Nd4j.create(shape, order);

        DataBuffer data = ret.data();
        for (long i = 0; i < length; i++) {
            data.put(i, nextDouble());
        }

        return ret;
    }

    @Override
    public INDArray nextFloat(long[] shape) {
        return nextFloat(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextFloat(int[] shape) {
        return nextFloat(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextFloat(char order, int[] shape) {
        return nextFloat(order, ArrayUtil.toLongArray(shape));
    }

    @Override
    public INDArray nextFloat(char order, long[] shape) {
        long length = ArrayUtil.prodLong(shape);
        INDArray ret = Nd4j.create(shape, order);

        DataBuffer data = ret.data();
        for (long i = 0; i < length; i++) {
            data.put(i, nextFloat());
        }

        return ret;
    }

    @Override
    public INDArray nextInt(int[] shape) {
        return nextInt(ArrayUtil.toLongArray(shape));
    }

    @Override
    public INDArray nextInt(long[] shape) {
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
        return nextInt(n, ArrayUtil.toLongArray(shape));
    }


    @Override
    public INDArray nextInt(int n, long[] shape) {
        int length = ArrayUtil.prod(shape);
        INDArray ret = Nd4j.create(shape);

        DataBuffer data = ret.data();
        for (int i = 0; i < length; i++) {
            data.put(i, nextInt(n));
        }

        return ret;
    }


    public synchronized RandomGenerator getRandomGenerator() {
        return randomGenerator;
    }

    public synchronized long getSeed() {
        return this.seed;
    }


    /**
     * This method returns pointer to RNG state structure.
     * Please note: DefaultRandom implementation returns NULL here, making it impossible to use with RandomOps
     *
     * @return
     */
    @Override
    public Pointer getStatePointer() {
        return null;
    }

    /**
     * This method returns pointer to RNG buffer
     *
     * @return
     */
    @Override
    public DataBuffer getStateBuffer() {
        return null;
    }

    @Override
    public void close() throws Exception {
        //
    }

    /**
     * Identical to setSeed(System.currentTimeMillis());
     */
    @Override
    public void reSeed() {
        reSeed(System.currentTimeMillis());
    }

    /**
     * Identical to setSeed(seed);
     *
     * @param seed
     */
    @Override
    public void reSeed(long seed) {
        setSeed(seed);
    }
}
