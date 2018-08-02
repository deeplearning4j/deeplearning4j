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

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Random generation based on commons math.
 * This is mean to be an independent.
 *
 * @author Adam Gibson
 */
public interface Random extends AutoCloseable {

    /**
     * Sets the seed of the underlying random number generator using an
     * <code>int</code> seed.
     * <p>Sequences of values generated starting with the same seeds
     * should be identical.
     * </p>
     *
     * @param seed the seed value
     */
    void setSeed(int seed);

    /**
     * Sets the seed of the underlying random number generator using an
     * <code>int</code> seed.
     * <p>Sequences of values generated starting with the same seeds
     * should be identical.
     * </p>
     *
     * @param seed the seed value
     */
    void setSeed(int[] seed);


    /**
     * Sets the seed of the underlying random number generator using a
     * <code>long</code> seed.
     * <p>Sequences of values generated starting with the same seeds
     * should be identical.
     * </p>
     *
     * @param seed the seed value
     */
    void setSeed(long seed);

    /**
     * Gets the  <code>long</code> seed of the underlying
     * random number generator.
     *
     * @return the seed value
     */
    long getSeed();

    /**
     * Generates random bytes and places them into a user-supplied
     * byte array.  The number of random bytes produced is equal to
     * the length of the byte array.
     *
     * @param bytes the non-null byte array in which to put the
     *              random bytes
     */
    void nextBytes(byte[] bytes);

    /**
     * Returns the next pseudorandom, uniformly distributed <code>int</code>
     * value from this random number generator's sequence.
     * All 2<font size="-1"><sup>32</sup></font> possible <tt>int</tt> values
     * should be produced with  (approximately) equal probability.
     *
     * @return the next pseudorandom, uniformly distributed <code>int</code>
     * value from this random number generator's sequence
     */
    int nextInt();

    /**
     * Returns a pseudorandom, uniformly distributed <tt>int</tt> value
     * between 0 (inclusive) and the specified value (exclusive), drawn from
     * this random number generator's sequence.
     *
     * @param n the bound on the random number to be returned.  Must be
     *          positive.
     * @return a pseudorandom, uniformly distributed <tt>int</tt>
     * value between 0 (inclusive) and n (exclusive).
     * @throws IllegalArgumentException if n is not positive.
     */
    int nextInt(int n);

    /**
     * Returns the next pseudorandom, uniformly distributed <code>long</code>
     * value from this random number generator's sequence.  All
     * 2<font size="-1"><sup>64</sup></font> possible <tt>long</tt> values
     * should be produced with (approximately) equal probability.
     *
     * @return the next pseudorandom, uniformly distributed <code>long</code>
     * value from this random number generator's sequence
     */
    long nextLong();

    /**
     * Returns the next pseudorandom, uniformly distributed
     * <code>boolean</code> value from this random number generator's
     * sequence.
     *
     * @return the next pseudorandom, uniformly distributed
     * <code>boolean</code> value from this random number generator's
     * sequence
     */
    boolean nextBoolean();

    /**
     * Returns the next pseudorandom, uniformly distributed <code>float</code>
     * value between <code>0.0</code> and <code>1.0</code> from this random
     * number generator's sequence.
     *
     * @return the next pseudorandom, uniformly distributed <code>float</code>
     * value between <code>0.0</code> and <code>1.0</code> from this
     * random number generator's sequence
     */
    float nextFloat();

    /**
     * Returns the next pseudorandom, uniformly distributed
     * <code>double</code> value between <code>0.0</code> and
     * <code>1.0</code> from this random number generator's sequence.
     *
     * @return the next pseudorandom, uniformly distributed
     * <code>double</code> value between <code>0.0</code> and
     * <code>1.0</code> from this random number generator's sequence
     */
    double nextDouble();

    /**
     * Returns the next pseudorandom, Gaussian ("normally") distributed
     * <code>double</code> value with mean <code>0.0</code> and standard
     * deviation <code>1.0</code> from this random number generator's sequence.
     *
     * @return the next pseudorandom, Gaussian ("normally") distributed
     * <code>double</code> value with mean <code>0.0</code> and
     * standard deviation <code>1.0</code> from this random number
     * generator's sequence
     */
    double nextGaussian();

    /**
     * Generate a gaussian number ndarray
     * of the specified shape
     *
     * @param shape the shape to generate
     * @return the generated gaussian numbers
     */
    INDArray nextGaussian(int[] shape);


    INDArray nextGaussian(long[] shape);

    /**
     * Generate a gaussian number ndarray
     * of the specified shape and order
     *
     * @param order the order of the output array
     * @param shape the shape to generate
     * @return the generated gaussian numbers
     */
    INDArray nextGaussian(char order, int[] shape);


    INDArray nextGaussian(char order, long[] shape);

    /**
     * Generate a uniform number ndarray
     * of the specified shape
     *
     * @param shape the shape to generate
     * @return the generated gaussian numbers
     */
    INDArray nextDouble(int[] shape);

    INDArray nextDouble(long[] shape);

    /**
     * Generate a uniform number ndarray
     * of the specified shape and order
     *
     * @param order order of the output array
     * @param shape the shape to generate
     * @return the generated gaussian numbers
     */
    INDArray nextDouble(char order, int[] shape);


    INDArray nextDouble(char order, long[] shape);

    /**
     * Generate a uniform number ndarray
     * of the specified shape
     *
     * @param shape the shape to generate
     * @return the generated uniform numbers
     */
    INDArray nextFloat(int[] shape);


    INDArray nextFloat(long[] shape);

    /**
     * Generate a uniform number ndarray
     * of the specified shape
     *
     * @param shape the shape to generate
     * @return the generated uniform numbers
     */
    INDArray nextFloat(char order, int[] shape);

    INDArray nextFloat(char order, long[] shape);

    /**
     * Generate a random set of integers
     * of the specified shape. Note that
     * these integers will not actually be integers
     * but floats that happen to be whole numbers.
     * The reason for this is due to ints
     * having the same space usage as floats.
     * This also plays nice with blas.
     * <p/>
     * If the data opType is set to double,
     * then these will be whole doubles.
     *
     * @param shape the shape to generate
     * @return the ndarray with
     * the shape of only integers.
     */
    INDArray nextInt(int[] shape);

    INDArray nextInt(long[] shape);


    /**
     * Generate a random set of integers
     * of the specified shape. Note that
     * these integers will not actually be integers
     * but floats that happen to be whole numbers.
     * The reason for this is due to ints
     * having the same space usage as floats.
     * This also plays nice with blas.
     * <p/>
     * If the data opType is set to double,
     * then these will be whole doubles.
     *
     * @param shape the shape to generate
     * @param n     the max number to generate trod a
     * @return the ndarray with
     * the shape of only integers.
     */
    INDArray nextInt(int n, int[] shape);

    INDArray nextInt(int n, long[] shape);

    /**
     * This method returns pointer to RNG state structure.
     * Please note: DefaultRandom implementation returns NULL here, making it impossible to use with RandomOps
     *
     * @return
     */
    Pointer getStatePointer();

    /**
     * This method returns pointer to RNG buffer
     *
     * @return
     */
    DataBuffer getStateBuffer();

    /**
     * This method is similar to setSeed() but it doesn't really touches underlying buffer, if any. So it acts as additional modifier to current RNG state. System.currentTimeMillis() will be used as modifier.
     *
     * PLEASE NOTE: Never use this method unless you're 100% sure what it does and why you would need it.
     *
     */
    void reSeed();

    /**
     * This method is similar to setSeed() but it doesn't really touches underlying buffer, if any. So it acts as additional modifier to current RNG state.
     *
     * PLEASE NOTE: Never use this method unless you're 100% sure what it does and why you would need it.
     *
     * @param seed
     */
    void reSeed(long seed);
}


