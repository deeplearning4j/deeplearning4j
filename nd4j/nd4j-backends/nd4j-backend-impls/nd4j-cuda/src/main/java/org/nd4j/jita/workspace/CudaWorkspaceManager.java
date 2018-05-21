package org.nd4j.jita.workspace;

import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.memory.provider.BasicWorkspaceManager;

/**
 * @author raver119@gmail.com
 */
public class CudaWorkspaceManager extends BasicWorkspaceManager {

    public CudaWorkspaceManager(){
        super();
    }


    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration) {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CudaWorkspace(configuration);

        backingMap.get().put(workspace.getId(), workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace() {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CudaWorkspace(defaultConfiguration);

        backingMap.get().put(workspace.getId(), workspace);
        pickReference(workspace);

        return workspace;
    }


    @Override
    public MemoryWorkspace createNewWorkspace(WorkspaceConfiguration configuration, String id) {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CudaWorkspace(configuration, id);

        backingMap.get().put(id, workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(WorkspaceConfiguration configuration, String id, Integer deviceId) {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CudaWorkspace(configuration, id, deviceId);

        backingMap.get().put(id, workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread(@NonNull WorkspaceConfiguration configuration, @NonNull String id) {
        ensureThreadExistense();

        MemoryWorkspace workspace = backingMap.get().get(id);
        if (workspace == null) {
            workspace = new CudaWorkspace(configuration, id);
            backingMap.get().put(id, workspace);
            pickReference(workspace);
        }

        return workspace;
    }


}
