package org.nd4j.tensorflow.conversion;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.tensorflow;

public class DummyDeAllocator extends tensorflow.Deallocator_Pointer_long_Pointer {
    private static DummyDeAllocator INSTANCE = new DummyDeAllocator();

    public static DummyDeAllocator getInstance() {
        return INSTANCE;
    }

    @Override
    public void call(Pointer pointer, long l, Pointer pointer1) {
    }
}
