package org.nd4j.linalg.api.memory;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;

import javax.annotation.Nonnull;

/**
 * This interface describes backend-specific implementations of MemoryWorkspaceProvider, basically Factory + Thread-based provider
 *
 * @author raver119@gmail.com
 */
public interface MemoryWorkspaceProvider {
    /**
     * This method sets default workspace configuration for this provider instance
     *
     * @param configuration
     */
    void setDefaultWorkspaceConfiguration(WorkspaceConfiguration configuration);

    /**
     * This method builds new Workspace with given configuration
     *
     * @param configuration
     * @return
     */
    MemoryWorkspace createNewWorkspace(WorkspaceConfiguration configuration);

    /**
     * This method builds new Workspace with default configuration
     *
     * @return
     */
    MemoryWorkspace createNewWorkspace();

    /**
     * This method returns you current Workspace for current Thread
     *
     * PLEASE NOTE: If Workspace wasn't defined, new Workspace will be created using current default configuration
     *
     * @return
     */
    MemoryWorkspace getWorkspaceForCurrentThread();

    /**
     * This method allows you to set given Workspace as default for current Thread
     *
     * @param workspace
     */
    void setWorkspaceForCurrentThread(MemoryWorkspace workspace);

    /**
     * This method allows you to destroy given Workspace
     *
     * @param workspace
     */
    void destroyWorkspace(MemoryWorkspace workspace);

    /**
     * This method destroys current Workspace for current Thread
     */
    void destroyWorkspace();
}
