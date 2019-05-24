/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
