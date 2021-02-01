/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.support;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MockRandom implements org.nd4j.linalg.api.rng.Random {

    private int randomDoubleValuesIdx = 0;
    private final double[] randomDoubleValues;

    private int randomIntValuesIdx = 0;
    private final int[] randomIntValues;

    public MockRandom(double[] randomDoubleValues, int[] randomIntValues) {
        this.randomDoubleValues = randomDoubleValues;
        this.randomIntValues = randomIntValues;
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
    public long getSeed() {
        return 0;
    }

    @Override
    public void nextBytes(byte[] bytes) {

    }

    @Override
    public int nextInt() {
        return randomIntValues[randomIntValuesIdx++];
    }

    @Override
    public int nextInt(int i) {
        return randomIntValues[randomIntValuesIdx++];
    }

    @Override
    public int nextInt(int i, int i1) {
        return randomIntValues[randomIntValuesIdx++];
    }

    @Override
    public long nextLong() {
        return randomIntValues[randomIntValuesIdx++];
    }

    @Override
    public boolean nextBoolean() {
        return false;
    }

    @Override
    public float nextFloat() {
        return (float)randomDoubleValues[randomDoubleValuesIdx++];
    }

    @Override
    public double nextDouble() {
        return randomDoubleValues[randomDoubleValuesIdx++];
    }

    @Override
    public double nextGaussian() {
        return 0;
    }

    @Override
    public INDArray nextGaussian(int[] ints) {
        return null;
    }

    @Override
    public INDArray nextGaussian(long[] longs) {
        return null;
    }

    @Override
    public INDArray nextGaussian(char c, int[] ints) {
        return null;
    }

    @Override
    public INDArray nextGaussian(char c, long[] longs) {
        return null;
    }

    @Override
    public INDArray nextDouble(int[] ints) {
        return null;
    }

    @Override
    public INDArray nextDouble(long[] longs) {
        return null;
    }

    @Override
    public INDArray nextDouble(char c, int[] ints) {
        return null;
    }

    @Override
    public INDArray nextDouble(char c, long[] longs) {
        return null;
    }

    @Override
    public INDArray nextFloat(int[] ints) {
        return null;
    }

    @Override
    public INDArray nextFloat(long[] longs) {
        return null;
    }

    @Override
    public INDArray nextFloat(char c, int[] ints) {
        return null;
    }

    @Override
    public INDArray nextFloat(char c, long[] longs) {
        return null;
    }

    @Override
    public INDArray nextInt(int[] ints) {
        return null;
    }

    @Override
    public INDArray nextInt(long[] longs) {
        return null;
    }

    @Override
    public INDArray nextInt(int i, int[] ints) {
        return null;
    }

    @Override
    public INDArray nextInt(int i, long[] longs) {
        return null;
    }

    @Override
    public Pointer getStatePointer() {
        return null;
    }

    @Override
    public long getPosition() {
        return 0;
    }

    @Override
    public void reSeed() {

    }

    @Override
    public void reSeed(long l) {

    }

    @Override
    public long rootState() {
        return 0;
    }

    @Override
    public long nodeState() {
        return 0;
    }

    @Override
    public void setStates(long l, long l1) {

    }

    @Override
    public void close() throws Exception {

    }
}
