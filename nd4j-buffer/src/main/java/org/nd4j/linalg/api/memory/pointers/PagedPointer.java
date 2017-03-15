package org.nd4j.linalg.api.memory.pointers;

import org.bytedeco.javacpp.*;

/**
 * @author raver119@gmail.com
 */
public class PagedPointer extends Pointer {

    // we're storing this pointer as strong reference
    private Pointer originalPointer;

    private PagedPointer() {

    }

    public PagedPointer(Pointer pointer) {
        this.originalPointer = pointer;
        this.address = pointer.address();
        this.capacity = pointer.capacity();
        this.limit = pointer.limit();
        this.position = 0;
    }

    public PagedPointer(Pointer pointer, long capacity) {
        this.address = pointer.address();
        this.capacity = capacity;
        this.limit = capacity;
        this.position = 0;
    }

    public PagedPointer(Pointer pointer, long capacity, long offset) {
        this.address = pointer.address() + offset;
        this.capacity = capacity;
        this.limit = capacity;
        this.position = 0;
    }


    public PagedPointer withOffset(long offset, long capacity) {
        return new PagedPointer(this, capacity, offset);
    }


    public FloatPointer asFloatPointer() {
        return new FloatPointer(this);
    }

    public DoublePointer asDoublePointer() {
        return new DoublePointer(this);
    }

    public IntPointer asIntPointer() {
        return new IntPointer(this);
    }

    public LongPointer asLongPointer() {
        return new LongPointer(this);
    }
}
