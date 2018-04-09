package org.nd4j.linalg.workspace;

import lombok.AllArgsConstructor;
import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class BaseWorkspaceMgr<T extends Enum<T>> implements WorkspaceMgr<T> {

    protected final Set<T> scopeOutOfWs = new HashSet<>();
    protected final Map<T, WorkspaceConfiguration> configMap = new HashMap<>();
    protected final Map<T, String> workspaceNames = new HashMap<>();

    @Override
    public void setConfiguration(@NonNull T arrayType, WorkspaceConfiguration configuration) {
        configMap.put(arrayType, configuration);
    }

    @Override
    public WorkspaceConfiguration getConfiguration(@NonNull T arrayType) {
        return configMap.get(arrayType);
    }

    @Override
    public void setScopedOutFor(@NonNull T arrayType) {
        scopeOutOfWs.add(arrayType);
        configMap.remove(arrayType);
        workspaceNames.remove(arrayType);
    }

    @Override
    public boolean isScopedOut(@NonNull T arrayType) {
        return scopeOutOfWs.contains(arrayType);
    }

    @Override
    public MemoryWorkspace notifyScopeEntered(@NonNull T arrayType) {
        validateConfig(arrayType);

        if(isScopedOut(arrayType)){
            return Nd4j.getWorkspaceManager().scopeOutOfWorkspaces();
        } else {
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                    getConfiguration(arrayType), getWorkspaceName(arrayType));
            return ws.notifyScopeEntered();
        }
    }

    @Override
    public AutoCloseable notifyScopeEntered(@NonNull T... arrayTypes) {
        MemoryWorkspace[] ws = new MemoryWorkspace[arrayTypes.length];
        for(int i=0; i<arrayTypes.length; i++ ){
            ws[i] = notifyScopeEntered(arrayTypes[i]);
        }
        return new WorkspacesCloseable(ws);
    }

    @Override
    public MemoryWorkspace notifyScopeBorrowed(@NonNull T arrayType) {
        validateConfig(arrayType);

        MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                getConfiguration(arrayType), getWorkspaceName(arrayType));
        return ws.notifyScopeBorrowed();
    }

    @Override
    public void setWorkspaceName(@NonNull T arrayType, @NonNull String name) {
        workspaceNames.put(arrayType, name);
    }

    @Override
    public String getWorkspaceName(@NonNull T arrayType) {
        return workspaceNames.get(arrayType);
    }

    @Override
    public void setWorkspace(@NonNull T forEnum, @NonNull String wsName, @NonNull WorkspaceConfiguration configuration) {
        setWorkspaceName(forEnum, wsName);
        setConfiguration(forEnum, configuration);
    }

    @Override
    public INDArray leverageTo(T arrayType, INDArray array) {
        if(array == null){
            return null;
        }
        validateConfig(arrayType);
        return array.leverageTo(getWorkspaceName(arrayType));
    }

    @Override
    public INDArray create(@NonNull T arrayType, @NonNull int... shape) {
        return create(arrayType, shape, Nd4j.order());
    }

    @Override
    public INDArray create(@NonNull T arrayType, @NonNull int[] shape, @NonNull char order) {
        try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)){
            return Nd4j.create(shape, order);
        }
    }

    @Override
    public INDArray createUninitialized(@NonNull T arrayType, @NonNull int... shape) {
        return createUninitialized(arrayType, shape, Nd4j.order());
    }

    @Override
    public INDArray createUninitialized(@NonNull T arrayType, @NonNull int[] shape, char order) {
        try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)){
            return Nd4j.createUninitialized(shape, order);
        }
    }


    private void validateConfig(@NonNull T arrayType){
        if(scopeOutOfWs.contains(arrayType)){
            return;
        }

        if(!configMap.containsKey(arrayType)){
            throw new IllegalStateException("No workspace configuration has been provided for arrayType: " + arrayType);
        }
        if(!workspaceNames.containsKey(arrayType)){
            throw new IllegalStateException("No workspace name has been provided for arrayType: " + arrayType);
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
