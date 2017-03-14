package org.nd4j.linalg.memory;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;

/**
 * This interface describes backend-specific implementations
 *
 * @author raver119@gmail.com
 */
public interface MemoryWorkspaceProvider {
    MemoryWorkspace createNewWorkspace(WorkspaceConfiguration configuration);

    void destroyWorkspace(MemoryWorkspace workspace);
}
