package org.nd4j.linalg.jcublas.buffer;

import jcuda.Pointer;

/**
 * Provides information about a device pointer
 *
 * @author bam4d
 */
public class DevicePointerInfo {
    final private Pointer pointer;
    final private long length;
    final private int stride;
    final private int offset;
    private boolean freed = false;

    public DevicePointerInfo(Pointer pointer, long length, int stride, int offset) {
        this.pointer = pointer;
        this.length = length;
        this.stride = stride;
        this.offset = offset;
    }

    public boolean isFreed() {
        return freed;
    }

    public void setFreed(boolean freed) {
        this.freed = freed;
    }

    public int getOffset() {
        return offset;
    }



    public int getStride() {
        return stride;
    }

    public Pointer getPointer() {
        return pointer;
    }

    public long getLength() {
        return length;
    }
}
