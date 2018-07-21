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
        this(seed, 10000000);
    }

    public CudaNativeRandom(long seed, long numberOfElements) {
        super(seed, numberOfElements);
    }

    @Override
    public void init() {
        statePointer = nativeOps.initRandom(getExtraPointers(), seed, numberOfElements,
                        AtomicAllocator.getInstance().getPointer(stateBuffer));

        AtomicAllocator.getInstance().getAllocationPoint(stateBuffer).tickDeviceWrite();
    }

    @Override
    public PointerPointer getExtraPointers() {
        PointerPointer ptr = new PointerPointer(4);
        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
        ptr.put(0, AtomicAllocator.getInstance().getHostPointer(stateBuffer));
        ptr.put(1, context.getOldStream());
        return ptr;
    }
}
