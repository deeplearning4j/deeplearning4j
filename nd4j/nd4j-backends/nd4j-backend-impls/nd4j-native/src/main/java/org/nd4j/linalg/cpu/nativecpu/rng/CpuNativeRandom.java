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

package org.nd4j.linalg.cpu.nativecpu.rng;

import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.nativeblas.Nd4jCpu;
import org.nd4j.rng.NativeRandom;

import java.util.concurrent.atomic.AtomicLong;

/**
 * CPU implementation for NativeRandom
 *
 * @author raver119@gmail.com
 */
public class CpuNativeRandom extends NativeRandom {
    public CpuNativeRandom() {
        super();
    }

    public CpuNativeRandom(long seed) {
        super(seed);
    }

    public CpuNativeRandom(long seed, long numberOfElements) {
        super(seed, numberOfElements);
    }

    @Override
    public void init() {
        statePointer = new Nd4jCpu.RandomGenerator(this.seed, this.seed ^ 0xdeadbeef);
    }

    @Override
    public PointerPointer getExtraPointers() {
        return null;
    }

    @Override
    public void setSeed(long seed) {
        this.seed = seed;
        this.currentPosition.set(0);
        ((Nd4jCpu.RandomGenerator)statePointer).setStates(seed, seed ^ 0xdeadbeef);
    }

    @Override
    public long getSeed() {
        return seed;
    }

    @Override
    public int nextInt() {
        return ((Nd4jCpu.RandomGenerator)statePointer).relativeInt(currentPosition.getAndIncrement());
    }

    @Override
    public long nextLong() {
        return ((Nd4jCpu.RandomGenerator)statePointer).relativeLong(currentPosition.getAndIncrement());
    }

    public long rootState() {
        return ((Nd4jCpu.RandomGenerator) statePointer).rootState();
    }

    public long nodeState() {
        return ((Nd4jCpu.RandomGenerator) statePointer).nodeState();
    }

    @Override
    public void setStates(long rootState, long nodeState) {
        ((Nd4jCpu.RandomGenerator) statePointer).setStates(rootState, nodeState);
    }
}
