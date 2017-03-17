package org.nd4j.linalg.memory.provider;

import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MemoryWorkspaceManager;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import java.util.HashMap;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
public class BasicWorkspaceManager implements MemoryWorkspaceManager {

    protected WorkspaceConfiguration defaultConfiguration;
    protected ThreadLocal<Map<String, MemoryWorkspace>> backingMap = new ThreadLocal<>();

    public BasicWorkspaceManager() {
        this(WorkspaceConfiguration.builder().initialSize(0).maxSize(0).overallocationLimit(0.3).policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build());
    }

    public BasicWorkspaceManager(@NonNull WorkspaceConfiguration defaultConfiguration) {
        this.defaultConfiguration = defaultConfiguration;
    }

    @Override
    public void setDefaultWorkspaceConfiguration(@NonNull WorkspaceConfiguration configuration) {
        this.defaultConfiguration = configuration;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration) {
        return new Nd4jWorkspace(configuration);
    }

    @Override
    public MemoryWorkspace createNewWorkspace() {
        return new Nd4jWorkspace(defaultConfiguration);
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread() {
        return getWorkspaceForCurrentThread(MemoryWorkspace.DEFAULT_ID);
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread(@NonNull String id) {
        return getWorkspaceForCurrentThread(defaultConfiguration, id);
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread(@NonNull WorkspaceConfiguration configuration, @NonNull String id) {
        ensureThreadExistense();

        MemoryWorkspace workspace = backingMap.get().get(id);
        if (workspace == null) {
            workspace = new Nd4jWorkspace(configuration, id);
            backingMap.get().put(id, workspace);
        }

        return workspace;
    }

    @Override
    public void setWorkspaceForCurrentThread(MemoryWorkspace workspace) {
        setWorkspaceForCurrentThread(workspace, MemoryWorkspace.DEFAULT_ID);
    }

    @Override
    public void setWorkspaceForCurrentThread(@NonNull MemoryWorkspace workspace, @NonNull String id) {
        ensureThreadExistense();

        backingMap.get().put(id, workspace);
    }

    @Override
    public void destroyWorkspace(@NonNull MemoryWorkspace workspace) {
        workspace.destroyWorkspace();
    }

    @Override
    public void destroyWorkspace() {
        ensureThreadExistense();

        MemoryWorkspace workspace = backingMap.get().get(MemoryWorkspace.DEFAULT_ID);
        if (workspace != null)
            workspace.destroyWorkspace();

        backingMap.get().remove(MemoryWorkspace.DEFAULT_ID);
    }

    @Override
    public void destroyAllWorkspacesForCurrentThread() {
        ensureThreadExistense();

        for (String key : backingMap.get().keySet()) {
            backingMap.get().get(key).destroyWorkspace();
        }
    }

    private void ensureThreadExistense() {
        if (backingMap.get() == null)
            backingMap.set(new HashMap<String, MemoryWorkspace>());
    }
}
