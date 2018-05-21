package org.nd4j.linalg.api.memory.enums;

/**
 * @author raver119@gmail.com
 */
public enum ResetPolicy {
    /**
     * This policy means - once end of MemoryWorkspace block reached its end - it'll be reset to the beginning.
     */
    BLOCK_LEFT,

    /**
     * This policy means - this Workspace instance will be acting as
     * circular buffer, so it'll be reset only after
     * end of workspace buffer is reached.
     */
    ENDOFBUFFER_REACHED,
}
