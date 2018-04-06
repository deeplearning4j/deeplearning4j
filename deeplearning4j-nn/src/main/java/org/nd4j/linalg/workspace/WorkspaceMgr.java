package org.nd4j.linalg.workspace;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface WorkspaceMgr<T extends Enum<T>> {

    void setConfiguration(T workspace, WorkspaceConfiguration configuration);

    WorkspaceConfiguration getConfiguration(T arrayType);

    void setScopedOutFor(T arrayType);

    boolean isScopedOut(T arrayType);

    MemoryWorkspace notifyScopeEntered(T arrayType);

    AutoCloseable notifyScopeEntered(T... arrayTypes);

    MemoryWorkspace notifyScopeBorrowed(T workspace);

    void setWorkspaceName(T arrayType, String wsName);

    String getWorkspaceName(T arrayType);

    void setWorkspace(T arrayType, String wsName, WorkspaceConfiguration configuration);

    INDArray leverageTo(T toWorkspace, INDArray array);

    INDArray create(T workspace, int... shape);

    INDArray create(T workspace, int[] shape, char ordering);

    INDArray createUninitialized(T workspace, int... shape);

    INDArray createUninitialized(T workspace, int[] shape, char order);



}
