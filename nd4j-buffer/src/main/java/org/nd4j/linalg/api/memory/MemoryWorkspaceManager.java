package org.nd4j.linalg.api.memory;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;

import edu.umd.cs.findbugs.annotations.NonNull;

import java.util.List;


/**
 * This interface describes backend-specific implementations of MemoryWorkspaceManager, basically Factory + Thread-based provider
 *
 * @author raver119@gmail.com
 */
public interface MemoryWorkspaceManager {

    /**
     * Returns globally unique ID
     *
     * @return
     */
    String getUUID();

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
     * This method builds new Workspace with given configuration
     *
     * @param configuration
     * @return
     */
    MemoryWorkspace createNewWorkspace(WorkspaceConfiguration configuration, String id);


    /**
     * This method builds new Workspace with given configuration
     *
     * @param configuration
     * @return
     */
    MemoryWorkspace createNewWorkspace(WorkspaceConfiguration configuration, String id, Integer deviceId);

    /**
     * This method returns you current default Workspace for current Thread
     *
     * PLEASE NOTE: If Workspace wasn't defined, new Workspace will be created using current default configuration
     *
     * @return
     */
    MemoryWorkspace getWorkspaceForCurrentThread();

    /**
     * This method returns you Workspace for a given Id for current Thread
     *
     * PLEASE NOTE: If Workspace wasn't defined, new Workspace will be created using current default configuration
     *
     * @return
     */
    MemoryWorkspace getWorkspaceForCurrentThread(String id);

    /**
     * This method returns you Workspace for a given Id for current Thread
     *
     * PLEASE NOTE: If Workspace wasn't defined, new Workspace will be created using given configuration
     *
     * @return
     */
    MemoryWorkspace getWorkspaceForCurrentThread(WorkspaceConfiguration configuration, String id);

    /**
     * This method allows you to set given Workspace as default for current Thread
     *
     * @param workspace
     */
    void setWorkspaceForCurrentThread(MemoryWorkspace workspace);

    /**
     * This method allows you to set given Workspace for spacific Id for current Thread
     *
     * @param workspace
     */
    void setWorkspaceForCurrentThread(MemoryWorkspace workspace, String id);

    /**
     * This method allows you to destroy given Workspace
     *
     * @param workspace
     */
    void destroyWorkspace(MemoryWorkspace workspace);

    /**
     * This method destroys & deallocates all Workspaces for a calling Thread
     *
     * PLEASE NOTE: This method is NOT safe
     */
    void destroyAllWorkspacesForCurrentThread();

    /**
     * This method destroys current Workspace for current Thread
     */
    void destroyWorkspace();

    /**
     * This method gets & activates default workspace
     *
     * @return
     */
    MemoryWorkspace getAndActivateWorkspace();

    /**
     * This method gets & activates workspace with a given Id
     *
     * @return
     */
    MemoryWorkspace getAndActivateWorkspace(String id);

    /**
     * This method gets & activates default with a given configuration and Id
     *
     * @return
     */
    MemoryWorkspace getAndActivateWorkspace(WorkspaceConfiguration configuration, String id);

    /**
     * This method checks, if Workspace with a given Id was created before this call
     *
     * @param id
     * @return
     */
    boolean checkIfWorkspaceExists(String id);

    /**
     * This method checks, if Workspace with a given Id was created before this call, AND is active at the moment of call
     *
     * @param id
     * @return
     */
    boolean checkIfWorkspaceExistsAndActive(String id);

    /**
     * This method temporary opens block out of any workspace scope.
     *
     * PLEASE NOTE: Do not forget to close this block.
     *
     * @return
     */
    MemoryWorkspace scopeOutOfWorkspaces();

    /**
     * This method prints out allocation statistics for current thread
     */
    void printAllocationStatisticsForCurrentThread();

    /**
     * This method returns list of workspace IDs for current thread
     *
     * @return
     */
    List<String> getAllWorkspacesIdsForCurrentThread();

    /**
     * This method returns all workspaces for current thread
     */
    List<MemoryWorkspace> getAllWorkspacesForCurrentThread();

    /**
     * Determine if there are any workspaces open for the current thread.
     *
     * @return True if any workspaces are open for this thread, false otherwise
     */
    boolean anyWorkspaceActiveForCurrentThread();
}
