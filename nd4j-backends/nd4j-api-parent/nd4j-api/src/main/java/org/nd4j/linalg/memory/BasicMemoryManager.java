package org.nd4j.linalg.memory;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class BasicMemoryManager implements MemoryManager {
    protected int frequency = 5;
    protected AtomicLong freqCounter = new AtomicLong(0);

    protected AtomicLong lastGcTime = new AtomicLong(0);

    /**
     * This method returns
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param bytes
     * @param kind
     * @param initialize
     */
    @Override
    public Pointer allocate(long bytes, MemoryKind kind, boolean initialize) {
        return null;
    }

    /**
     * This method detaches off-heap memory from passed INDArray instances, and optionally stores them in cache for future reuse
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param arrays
     */
    @Override
    public void collect(INDArray... arrays) {
        throw new UnsupportedOperationException("This method isn't implemented yet");
    }

    /**
     * This method purges all cached memory chunks
     *
     */
    @Override
    public void purgeCaches() {
        throw new UnsupportedOperationException("This method isn't implemented yet");
    }

    @Override
    public void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer) {
        Pointer.memcpy(dstBuffer.addressPointer(), srcBuffer.addressPointer(),
                        srcBuffer.length() * srcBuffer.getElementSize());
    }

    @Override
    public void notifyScopeEntered() {
        // TODO: to be implemented
    }

    @Override
    public void notifyScopeLeft() {
        // TODO: to be implemented
    }

    @Override
    public void invokeGcOccasionally() {
        if (frequency > 0)
            if (freqCounter.incrementAndGet() % frequency == 0) {
                System.gc();
                lastGcTime.set(System.currentTimeMillis());
            }
    }

    @Override
    public void invokeGc() {
        System.gc();
        lastGcTime.set(System.currentTimeMillis());
    }

    @Override
    public void setManualGcFrequency(int frequency) {
        this.frequency = frequency;
    }

    @Override
    public void setAutoGcWindow(long windowMillis) {
        //
    }

    @Override
    public long getLastGcTime() {
        return lastGcTime.get();
    }
}
