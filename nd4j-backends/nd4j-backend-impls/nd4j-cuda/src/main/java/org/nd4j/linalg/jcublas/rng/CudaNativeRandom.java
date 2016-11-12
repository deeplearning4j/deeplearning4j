package org.nd4j.linalg.jcublas.rng;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
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
        statePointer = nativeOps.initRandom(seed, numberOfElements, AtomicAllocator.getInstance().getHostPointer(stateBuffer));

        AtomicAllocator.getInstance().getAllocationPoint(stateBuffer).tickHostWrite();
    }


}
