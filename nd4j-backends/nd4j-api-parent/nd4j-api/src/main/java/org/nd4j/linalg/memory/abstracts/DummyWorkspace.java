package org.nd4j.linalg.memory.abstracts;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
public class DummyWorkspace implements MemoryWorkspace {
    /**
     * This method returns WorkspaceConfiguration bean that was used for given Workspace instance
     *
     * @return
     */
    @Override
    public WorkspaceConfiguration getWorkspaceConfiguration() {
        return null;
    }

    /**
     * This method returns Id of this workspace
     *
     * @return
     */
    @Override
    public String getId() {
        return null;
    }

    /**
     * This method does allocation from a given Workspace
     *
     * @param requiredMemory allocation size, in bytes
     * @param dataType       dataType that is going to be used
     * @param initialize
     * @return
     */
    @Override
    public PagedPointer alloc(long requiredMemory, DataBuffer.Type dataType, boolean initialize) {
        return null;
    }

    /**
     * This method does allocation from a given Workspace
     *
     * @param requiredMemory allocation size, in bytes
     * @param kind           MemoryKind for allocation
     * @param dataType       dataType that is going to be used
     * @param initialize
     * @return
     */
    @Override
    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type dataType, boolean initialize) {
        return null;
    }

    /**
     * This method notifies given Workspace that new use cycle is starting now
     *
     * @return
     */
    @Override
    public MemoryWorkspace notifyScopeEntered() {
        return null;
    }

    /**
     * This method notifies given Workspace that use cycle just ended
     *
     * @return
     */
    @Override
    public MemoryWorkspace notifyScopeLeft() {
        return null;
    }

    /**
     * This method returns True if scope was opened, and not closed yet.
     *
     * @return
     */
    @Override
    public boolean isScopeActive() {
        return false;
    }

    /**
     * This method causes Workspace initialization
     * <p>
     * PLEASE NOTE: This call will have no effect on previously initialized Workspace
     */
    @Override
    public void initializeWorkspace() {

    }

    /**
     * This method causes Workspace destruction: all memory allocations are released after this call.
     */
    @Override
    public void destroyWorkspace() {

    }

    /**
     * This method allows you to temporary disable/enable given Workspace use.
     * If turned off - direct memory allocations will be used.
     *
     * @param isEnabled
     */
    @Override
    public void toggleWorkspaceUse(boolean isEnabled) {

    }

    /**
     * This method returns amount of memory consumed in last successful cycle, in bytes
     *
     * @return
     */
    @Override
    public long getLastCycleAllocations() {
        return 0;
    }

    /**
     * This method returns amount of memory consumed by largest successful cycle, in bytes
     *
     * @return
     */
    @Override
    public long getMaxCycleAllocations() {
        return 0;
    }

    @Override
    public void close() {
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
    }

    /**
     * This method returns parent Workspace, if any. Null if there's none.
     *
     * @return
     */
    @Override
    public MemoryWorkspace getParentWorkspace() {
        return null;
    }

    @Override
    public MemoryWorkspace tagOutOfScopeUse() {
        return this;
    }
}
