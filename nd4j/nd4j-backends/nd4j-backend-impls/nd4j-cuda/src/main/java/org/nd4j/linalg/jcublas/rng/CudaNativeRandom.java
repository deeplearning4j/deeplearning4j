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

package org.nd4j.linalg.jcublas.rng;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.Nd4jCuda;
import org.nd4j.rng.NativeRandom;

import java.util.List;

/**
 * NativeRandom wrapper for CUDA with multi-gpu support
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CudaNativeRandom extends NativeRandom {

    protected List<DataBuffer> stateBuffers;

    public CudaNativeRandom() {
        this(System.currentTimeMillis());
    }

    public CudaNativeRandom(long seed) {
        super(seed);
    }

    public CudaNativeRandom(long seed, long nodeSeed) {
        super(seed, nodeSeed);
    }

    @Override
    public void init() {
        statePointer = new Nd4jCuda.RandomGenerator(seed, seed ^ 0xdeadbeef);
        setSeed(seed);
    }

    @Override
    public PointerPointer getExtraPointers() {
        return null;
    }

    @Override
    public void setSeed(long seed) {
        this.seed = seed;
        this.currentPosition.set(0);
        ((Nd4jCuda.RandomGenerator) statePointer).setStates(seed, seed ^ 0xdeadbeef);
    }

    @Override
    public long getSeed() {
        return seed;
    }

    @Override
    public int nextInt() {
        return ((Nd4jCuda.RandomGenerator) statePointer).relativeInt(currentPosition.getAndIncrement());
    }

    @Override
    public long nextLong() {
        return ((Nd4jCuda.RandomGenerator) statePointer).relativeLong(currentPosition.getAndIncrement());
    }

    public long rootState() {
        return ((Nd4jCuda.RandomGenerator) statePointer).rootState();
    }

    public long nodeState() {
        return ((Nd4jCuda.RandomGenerator) statePointer).nodeState();
    }
}
