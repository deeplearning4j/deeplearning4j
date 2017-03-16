package org.nd4j.linalg.api.memory;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;

/**
 * @author raver119@gmail.com
 */
public interface MemoryWorkspace extends AutoCloseable {

    /**
     * This method returns WorkspaceConfiguration bean that was used for given Workspace instance
     *
     * @return
     */
    WorkspaceConfiguration getWorkspaceConfiguration();

    /**
     * This method does allocation from a given Workspace
     *
     * @param requiredMemory allocation size, in bytes
     * @param dataType dataType that is going to be used
     * @return
     */
    PagedPointer alloc(long requiredMemory, DataBuffer.Type dataType);

    /**
     * This method does allocation from a given Workspace
     *
     * @param requiredMemory allocation size, in bytes
     * @param kind MemoryKind for allocation
     * @param dataType dataType that is going to be used
     * @return
     */
    PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type dataType);

    /**
     * This method notifies given Workspace that new use cycle is starting now
     *
     * @return
     */
    MemoryWorkspace notifyScopeEntered();

    /**
     * This method notifies given Workspace that use cycle just ended
     *
     * @return
     */
    MemoryWorkspace notifyScopeLeft();

    /**
     * This method causes Workspace initialization
     *
     * PLEASE NOTE: This call will have no effect on previously initialized Workspace
     */
    void initializeWorkspace();

    /**
     * This method causes Workspace destruction: all memory allocations are released after this call.
     */
    void destroyWorkspace();

    /**
     * This method allows you to temporary disable/enable given Workspace use.
     * If turned off - direct memory allocations will be used.
     *
     * @param isEnabled
     */
    void toggleWorkspaceUse(boolean isEnabled);

    /**
     * This method returns amount of memory consumed in last successful cycle, in bytes
     *
     * @return
     */
    long getLastCycleAllocations();

    /**
     * This method returns amount of memory consumed by largest successful cycle, in bytes
     * @return
     */
    long getMaxCycleAllocations();
}
