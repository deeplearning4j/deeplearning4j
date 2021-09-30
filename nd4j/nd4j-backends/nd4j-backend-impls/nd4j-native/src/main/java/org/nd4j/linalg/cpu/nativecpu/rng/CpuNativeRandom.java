/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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
        return nativeOps.getRandomGeneratorNextInt((OpaqueRandomGenerator)statePointer);
    }

    @Override
    public float nextFloat() {
        return nativeOps.getRandomGeneratorNextFloat((OpaqueRandomGenerator)statePointer);
    }

    @Override
    public double nextDouble() {
        return nativeOps.getRandomGeneratorNextDouble((OpaqueRandomGenerator)statePointer);
    }

    @Override
    public long nextLong() {
        return nativeOps.getRandomGeneratorNextLong((OpaqueRandomGenerator)statePointer);
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
