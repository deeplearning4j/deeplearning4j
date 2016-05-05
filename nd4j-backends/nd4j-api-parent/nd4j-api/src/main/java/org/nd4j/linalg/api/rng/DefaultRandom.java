/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.rng;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Apache commons based random number generation
 *
 * @author Adam Gibson
 */
public class DefaultRandom implements Random, RandomGenerator {
    protected RandomGenerator randomGenerator;
    protected long seed;
    /**
     * Initialize with a System.currentTimeMillis()
     * seed
     */
    public DefaultRandom() { this(System.currentTimeMillis()); }

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
        this.seed= seed;
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
    public INDArray nextGaussian(char order, int[] shape){
        INDArray ret = Nd4j.create(shape,order);
        INDArray linear = ret.linearView();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, nextGaussian());
        }
        return ret;
    }

    @Override
    public INDArray nextDouble(int[] shape) {
        return nextDouble(Nd4j.order(), shape);
    }

    @Override
    public INDArray nextDouble(char order, int[] shape){
        INDArray ret = Nd4j.create(shape,order);
        INDArray linear = ret.linearView();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, nextDouble());
        }
        return ret;
    }

    @Override
    public INDArray nextFloat(int[] shape) {
        INDArray ret = Nd4j.create(shape);
        INDArray linear = ret.linearView();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, nextFloat());
        }
        return ret;
    }

    @Override
    public INDArray nextInt(int[] shape) {
        INDArray ret = Nd4j.create(shape);
        INDArray linear = ret.linearView();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, nextInt());
        }
        return ret;
    }

    @Override
    public INDArray nextInt(int n, int[] shape) {
        INDArray ret = Nd4j.create(shape);
        INDArray linear = ret.linearView();
        for (int i = 0; i < linear.length(); i++) {
            linear.putScalar(i, nextInt(n));
        }
        return ret;
    }


    public synchronized RandomGenerator getRandomGenerator() {
        return randomGenerator;
    }

    public synchronized long getSeed(){
        return this.seed;
    }

}
