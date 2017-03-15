package org.nd4j.linalg.api.memory;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;

/**
 * @author raver119@gmail.com
 */
public interface MemoryWorkspace {

    WorkspaceConfiguration getWorkspaceConfiguration();

    PagedPointer alloc(long requiredMemory, DataBuffer.Type dataType);

    PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type dataType);
}
