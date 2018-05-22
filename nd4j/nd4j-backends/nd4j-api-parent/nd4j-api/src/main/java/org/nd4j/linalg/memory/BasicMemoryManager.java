package org.nd4j.linalg.memory;

import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public abstract class BasicMemoryManager implements MemoryManager {
    protected AtomicInteger frequency = new AtomicInteger(0);
    protected AtomicLong freqCounter = new AtomicLong(0);

    protected AtomicLong lastGcTime = new AtomicLong(System.currentTimeMillis());

    protected AtomicBoolean periodicEnabled = new AtomicBoolean(true);

    protected AtomicInteger averageLoopTime = new AtomicInteger(0);

    protected AtomicInteger noGcWindow = new AtomicInteger(100);

    protected AtomicBoolean averagingEnabled = new AtomicBoolean(false);

    protected static final int intervalTail = 100;

    protected Queue<Integer> intervals = new ConcurrentLinkedQueue<>();

    private ThreadLocal<MemoryWorkspace> workspace = new ThreadLocal<>();

    private ThreadLocal<MemoryWorkspace> tempWorkspace = new ThreadLocal<>();


    /**
     * This method returns
     * PLEASE NOTE: Cache options
     * depend on specific implementations
     *
     * @param bytes
     * @param kind
     * @param initialize
     */
    @Override
    public Pointer allocate(long bytes, MemoryKind kind, boolean initialize) {
        throw new UnsupportedOperationException("This method isn't available for this backend");
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

    @Override
    public void toggleAveraging(boolean enabled) {
        averagingEnabled.set(enabled);
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
        val perfD = PerformanceTracker.getInstance().helperStartTransaction();

        Pointer.memcpy(dstBuffer.addressPointer(), srcBuffer.addressPointer(),
                        srcBuffer.length() * srcBuffer.getElementSize());

        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfD, srcBuffer.length() * srcBuffer.getElementSize(), MemcpyDirection.HOST_TO_HOST);
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
        long currentTime = System.currentTimeMillis();

        if (averagingEnabled.get())
            intervals.add((int) (currentTime - lastGcTime.get()));

        // not sure if we want to conform autoGcWindow here...
        if (frequency.get() > 0)
            if (freqCounter.incrementAndGet() % frequency.get() == 0
                            && currentTime > getLastGcTime() + getAutoGcWindow()) {
                System.gc();
                lastGcTime.set(System.currentTimeMillis());
            }

        if (averagingEnabled.get())
            if (intervals.size() > intervalTail)
                intervals.remove();
    }

    @Override
    public void invokeGc() {
        System.gc();
        lastGcTime.set(System.currentTimeMillis());
    }

    @Override
    public boolean isPeriodicGcActive() {
        return periodicEnabled.get();
    }

    @Override
    public void setOccasionalGcFrequency(int frequency) {
        this.frequency.set(frequency);
    }

    @Override
    public void setAutoGcWindow(int windowMillis) {
        noGcWindow.set(windowMillis);
    }

    @Override
    public int getAutoGcWindow() {
        return noGcWindow.get();
    }

    @Override
    public int getOccasionalGcFrequency() {
        return frequency.get();
    }

    @Override
    public long getLastGcTime() {
        return lastGcTime.get();
    }

    @Override
    public void togglePeriodicGc(boolean enabled) {
        periodicEnabled.set(enabled);
    }

    @Override
    public int getAverageLoopTime() {
        if (averagingEnabled.get()) {
            int cnt = 0;
            for (Integer value : intervals) {
                cnt += value;
            }
            cnt /= intervals.size();
            return cnt;
        } else
            return 0;

    }

    @Override
    public MemoryWorkspace getCurrentWorkspace() {
        return workspace.get();
    }

    @Override
    public void setCurrentWorkspace(MemoryWorkspace workspace) {
        this.workspace.set(workspace);
    }


    @Override
    public MemoryWorkspace scopeOutOfWorkspaces() {
        MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        if (workspace == null)
            return new DummyWorkspace();
        else {
            //Nd4j.getMemoryManager().setCurrentWorkspace(null);
            return new DummyWorkspace().notifyScopeEntered();//workspace.tagOutOfScopeUse();
        }
    }
}
