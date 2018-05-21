package org.nd4j.linalg.api.memory;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;

/**
 * This interface describes reusable memory chunks abstraction
 *
 * @author raver119@gmail.com
 */
public interface MemoryWorkspace extends AutoCloseable {
    String DEFAULT_ID = "DefaultWorkspace";

    enum Type {
        /**
         * This mode means you have dummy workspace here. It doesn't provide any functionality.
         */
        DUMMY,

        /**
         * Most regular workspace. It starts somewhere, and ends somewhere. It has limits, aka scope of use.
         */
        SCOPED,

        /**
         * Special workspace mode: circular buffer. Workspace is never closed, and gets reset only once end reached.
         */
        CIRCULAR,
    }

    /**
     * This method returns WorkspaceConfiguration bean that was used for given Workspace instance
     *
     * @return
     */
    WorkspaceConfiguration getWorkspaceConfiguration();

    /**
     * This method returns Type of this workspace
     *
     * @return
     */
    Type getWorkspaceType();

    /**
     * This method returns Id of this workspace
     *
     * @return
     */
    String getId();

    /**
     * Returns deviceId for this workspace
     *
     * @return
     */
    int getDeviceId();

    /**
     * This method returns threadId where this workspace was created
     *
     * @return
     */
    Long getThreadId();

    /**
     * This method returns current generation Id
     * @return
     */
    long getGenerationId();

    /**
     * This method does allocation from a given Workspace
     *
     * @param requiredMemory allocation size, in bytes
     * @param dataType dataType that is going to be used
     * @return
     */
    PagedPointer alloc(long requiredMemory, DataBuffer.Type dataType, boolean initialize);

    /**
     * This method does allocation from a given Workspace
     *
     * @param requiredMemory allocation size, in bytes
     * @param kind MemoryKind for allocation
     * @param dataType dataType that is going to be used
     * @return
     */
    PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type dataType, boolean initialize);

    /**
     * This method notifies given Workspace that new use cycle is starting now
     *
     * @return
     */
    MemoryWorkspace notifyScopeEntered();

    /**
     * This method TEMPORARY enters this workspace, without reset applied
     *
     * @return
     */
    MemoryWorkspace notifyScopeBorrowed();

    /**
     * This method notifies given Workspace that use cycle just ended
     *
     * @return
     */
    MemoryWorkspace notifyScopeLeft();

    /**
     * This method returns True if scope was opened, and not closed yet.
     *
     * @return
     */
    boolean isScopeActive();

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

    void destroyWorkspace(boolean extended);

    /**
     * This method allows you to temporary disable/enable given Workspace use.
     * If turned off - direct memory allocations will be used.
     *
     * @param isEnabled
     */
    void toggleWorkspaceUse(boolean isEnabled);

    /**
     * This method returns amount of memory consumed in current cycle, in bytes
     *
     * @return
     */
    long getThisCycleAllocations();

    /**
     * This method enabled debugging mode for this workspace
     *
     * @param reallyEnable
     */
    void enableDebug(boolean reallyEnable);

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

    /**
     * This methos returns current allocated size of this workspace
     *
     * @return
     */
    long getCurrentSize();

    /**
     * This method is for compatibility with "try-with-resources" java blocks.
     * Internally it should be equal to notifyScopeLeft() method
     *
     */
    @Override
    void close();

    /**
     * This method returns parent Workspace, if any. Null if there's none.
     * @return
     */
    MemoryWorkspace getParentWorkspace();

    /**
     * This method temporary disables this workspace
     *
     * @return
     */
    MemoryWorkspace tagOutOfScopeUse();

    /**
     * Set the previous workspace, if any<br>
     * NOTE: this method should only be used if you are fully aware of the consequences of doing so. Incorrect use
     * of this method may leave workspace management in an invalid/indeterminant state!
     *
     * @param memoryWorkspace Workspace to set as the previous workspace. This is the workspace that will become active
     *                        when this workspace is closed.
     */
    void setPreviousWorkspace(MemoryWorkspace memoryWorkspace);
}
