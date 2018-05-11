package org.nd4j.linalg.memory;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 *
 * @author raver119@gmail.com
 */
public interface MemoryManager {

    MemoryWorkspace getCurrentWorkspace();

    void setCurrentWorkspace(MemoryWorkspace workspace);

    /**
     * PLEASE NOTE: This method is under development yet. Do not use it.
     */
    void notifyScopeEntered();

    /**
     * PLEASE NOTE: This method is under development yet. Do not use it.
     */
    void notifyScopeLeft();

    /**
     * This method calls for GC, and if frequency is met - System.gc() will be called
     */
    void invokeGcOccasionally();

    /**
     * This method calls for GC.
     */
    void invokeGc();

    /**
     * This method enables/disables periodic GC
     *
     * @param enabled
     */
    void togglePeriodicGc(boolean enabled);

    /**
     * This method enables/disables calculation of average time spent within loops
     *
     * Default: false
     *
     * @param enabled
     */
    void toggleAveraging(boolean enabled);

    /**
     * This method returns true, if periodic GC is active. False otherwise.
     *
     * @return
     */
    boolean isPeriodicGcActive();

    /**
     * This method returns time (in milliseconds) of the las System.gc() call
     *
     * @return
     */
    long getLastGcTime();

    /**
     * Sets manual GC invocation frequency. If you set it to 5, only 1/5 of calls will result in GC invocation
     * If 0 is used as frequency, it'll disable all manual invocation hooks.
     *
     * default value: 5
     * @param frequency
     */
    void setOccasionalGcFrequency(int frequency);

    /**
     * This method returns
     * @return
     */
    int getOccasionalGcFrequency();

    /**
     * This method returns average time between invokeGCOccasionally() calls
     * @return
     */
    int getAverageLoopTime();

    /**
     * This method enables/disables periodic System.gc() calls.
     * Set to 0 to disable this option.
     *
     * @param windowMillis minimal time milliseconds between calls.
     */
    void setAutoGcWindow(int windowMillis);

    /**
     * This method reutrns
     */
    int getAutoGcWindow();

    /**
     * This method returns pointer to allocated memory
     *
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param bytes
     */
    Pointer allocate(long bytes, MemoryKind kind, boolean initialize);


    /**
     * This method releases previously allocated memory chunk
     *
     * @param pointer
     * @param kind
     * @return
     */
    void release(Pointer pointer, MemoryKind kind);

    /**
     * This method detaches off-heap memory from passed INDArray instances, and optionally stores them in cache for future reuse
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param arrays
     */
    void collect(INDArray... arrays);


    /**
     * This method purges all cached memory chunks
     * 
     */
    void purgeCaches();

    /**
     * This method does memcpy  from source buffer to destination buffer
     *
     * PLEASE NOTE: This method is NOT safe.
     *
     * @param dstBuffer
     * @param srcBuffer
     */
    void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer);


    /**
     * This method fills given INDArray with zeroes.
     *
     * PLEASE NOTE: Can't be efficiently used on views, .assign(0.0) will be used instead
     *
     * @param array
     */
    void memset(INDArray array);

    /**
     * This method temporary opens block out of any workspace scope.
     *
     * PLEASE NOTE: Do not forget to close this block.
     *
     * @return
     */
    MemoryWorkspace scopeOutOfWorkspaces();

    /**
     *  This method returns per-device bandwidth use for memory transfers
     */
    Map<Integer, Long> getBandwidthUse();
}
