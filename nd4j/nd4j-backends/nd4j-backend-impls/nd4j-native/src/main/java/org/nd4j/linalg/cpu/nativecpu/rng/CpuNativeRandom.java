package org.nd4j.linalg.cpu.nativecpu.rng;

import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.rng.NativeRandom;

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
        statePointer = nativeOps.initRandom(getExtraPointers(), seed, numberOfElements, stateBuffer.addressPointer());
    }

    @Override
    public PointerPointer getExtraPointers() {
        return null;
    }
}
