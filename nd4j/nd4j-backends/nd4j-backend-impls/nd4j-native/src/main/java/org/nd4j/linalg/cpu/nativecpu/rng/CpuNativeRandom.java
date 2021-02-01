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

package org.nd4j.linalg.cpu.nativecpu.rng;

import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueRandomGenerator;
import org.nd4j.rng.NativeRandom;

/**
 * CPU implementation for NativeRandom
 *
 * @author raver119@gmail.com
 */
public class CpuNativeRandom extends NativeRandom {
    private NativeOps nativeOps;

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
        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        statePointer = nativeOps.createRandomGenerator(this.seed, this.seed ^ 0xdeadbeef);
    }

    @Override
    public void close() {
        nativeOps.deleteRandomGenerator((OpaqueRandomGenerator)statePointer);
    }

    @Override
    public PointerPointer getExtraPointers() {
        return null;
    }

    @Override
    public void setSeed(long seed) {
        this.seed = seed;
        this.currentPosition.set(0);
        nativeOps.setRandomGeneratorStates((OpaqueRandomGenerator)statePointer, seed, seed ^ 0xdeadbeef);
    }

    @Override
    public long getSeed() {
        return seed;
    }

    @Override
    public int nextInt() {
        return nativeOps.getRandomGeneratorRelativeInt((OpaqueRandomGenerator)statePointer, currentPosition.getAndIncrement());
    }

    @Override
    public float nextFloat() {
        return nativeOps.getRandomGeneratorRelativeFloat((OpaqueRandomGenerator)statePointer, currentPosition.getAndIncrement());
    }

    @Override
    public double nextDouble() {
        return nativeOps.getRandomGeneratorRelativeDouble((OpaqueRandomGenerator)statePointer, currentPosition.getAndIncrement());
    }

    @Override
    public long nextLong() {
        return nativeOps.getRandomGeneratorRelativeLong((OpaqueRandomGenerator)statePointer, currentPosition.getAndIncrement());
    }

    public long rootState() {
        return nativeOps.getRandomGeneratorRootState((OpaqueRandomGenerator)statePointer);
    }

    public long nodeState() {
        return nativeOps.getRandomGeneratorNodeState((OpaqueRandomGenerator)statePointer);
    }

    @Override
    public void setStates(long rootState, long nodeState) {
        nativeOps.setRandomGeneratorStates((OpaqueRandomGenerator)statePointer, rootState, nodeState);
    }
}
