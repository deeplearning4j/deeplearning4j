package org.nd4j.nativeblas;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;

/**
 * Wrapper for DoublePointer -> LongPointer
 */
public class LongPointerWrapper extends LongPointer {

    public LongPointerWrapper(Pointer pointer) {
        this.address = pointer.address();
        this.capacity = pointer.capacity();
        this.limit = pointer.limit();
        this.position = pointer.position();
    }
}
