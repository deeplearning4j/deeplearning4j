package org.nd4j.linalg.cpu.nativecpu.workspace;

import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.memory.provider.BasicWorkspaceManager;

/**
 * @author raver119@gmail.com
 */
public class CpuWorkspaceManager extends BasicWorkspaceManager {

    public CpuWorkspaceManager() {
        super();
    }

    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration) {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CpuWorkspace(configuration);

        backingMap.get().put(workspace.getId(), workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace() {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CpuWorkspace(defaultConfiguration);

        backingMap.get().put(workspace.getId(), workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String id) {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CpuWorkspace(configuration, id);

        backingMap.get().put(id, workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String id, Integer deviceId) {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CpuWorkspace(configuration, id, deviceId);

        backingMap.get().put(id, workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread(@NonNull WorkspaceConfiguration configuration, @NonNull String id) {
        ensureThreadExistense();

        MemoryWorkspace workspace = backingMap.get().get(id);
        if (workspace == null) {
            workspace = new CpuWorkspace(configuration, id);
            backingMap.get().put(id, workspace);
            pickReference(workspace);
        }

        return workspace;
    }
}
