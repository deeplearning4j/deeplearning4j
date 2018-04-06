package org.nd4j.linalg.workspace;

import lombok.AllArgsConstructor;
import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

public class BaseWorkspaceMgr<T extends Enum<T>> implements WorkspaceMgr<T> {

    private final Map<T, WorkspaceConfiguration> configMap = new HashMap<>();
    private final Map<T, String> workspaceNames = new HashMap<>();

    @Override
    public void setConfiguration(@NonNull T workspace, WorkspaceConfiguration configuration) {
        configMap.put(workspace, configuration);
    }

    @Override
    public WorkspaceConfiguration getConfiguration(@NonNull T workspace) {
        return configMap.get(workspace);
    }

    @Override
    public MemoryWorkspace notifyScopeEntered(@NonNull T workspace) {
        validateConfig(workspace);

        MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                getConfiguration(workspace), getWorkspaceName(workspace));
        return ws.notifyScopeEntered();
    }

    @Override
    public AutoCloseable notifyScopeEntered(@NonNull T... workspaces) {
        MemoryWorkspace[] ws = new MemoryWorkspace[workspaces.length];
        for(int i=0; i<workspaces.length; i++ ){
            ws[i] = notifyScopeEntered(workspaces[i]);
        }
        return new WorkspacesCloseable(ws);
    }

    @Override
    public MemoryWorkspace notifyScopeBorrowed(@NonNull T workspace) {
        validateConfig(workspace);

        MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                getConfiguration(workspace), getWorkspaceName(workspace));
        return ws.notifyScopeBorrowed();
    }

    @Override
    public void setWorkspaceName(@NonNull T workspace, @NonNull String name) {
        workspaceNames.put(workspace, name);
    }

    @Override
    public String getWorkspaceName(@NonNull T workspace) {
        return workspaceNames.get(workspace);
    }

    @Override
    public INDArray create(@NonNull T workspace, @NonNull int... shape) {
        return create(workspace, shape, Nd4j.order());
    }

    @Override
    public INDArray create(@NonNull T workspace, @NonNull int[] shape, @NonNull char order) {
        try(MemoryWorkspace ws = notifyScopeBorrowed(workspace)){
            return Nd4j.create(shape, order);
        }
    }

    @Override
    public INDArray createUninitialized(@NonNull T workspace, @NonNull int... shape) {
        return createUninitialized(workspace, shape, Nd4j.order());
    }

    @Override
    public INDArray createUninitialized(@NonNull T workspace, @NonNull int[] shape, char order) {
        try(MemoryWorkspace ws = notifyScopeBorrowed(workspace)){
            return Nd4j.createUninitialized(shape, order);
        }
    }


    private void validateConfig(@NonNull T workspace){
        if(!configMap.containsKey(workspace)){
            throw new IllegalStateException("No workspace configuration has been provided for workspace: " + workspace);
        }
        if(!workspaceNames.containsKey(workspace)){
            throw new IllegalStateException("No workspace name has been provided for workspace: " + workspace);
        }
    }

    @AllArgsConstructor
    private static class WorkspacesCloseable implements AutoCloseable {
        private MemoryWorkspace[] workspaces;

        @Override
        public void close() throws Exception {
            for(MemoryWorkspace ws : workspaces){
                ws.close();
            }
        }
    }
}
