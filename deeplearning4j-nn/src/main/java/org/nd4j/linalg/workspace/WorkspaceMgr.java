package org.nd4j.linalg.workspace;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface WorkspaceMgr<T extends Enum<T>> {

    void setConfiguration(T workspace, WorkspaceConfiguration configuration);

    WorkspaceConfiguration getConfiguration(T workspace);

    void setDummyWorkspace(T workspace);

    boolean isDummyWorkspace(T workspace);

    MemoryWorkspace notifyScopeEntered(T workspace);

    AutoCloseable notifyScopeEntered(T... workspaces);

    MemoryWorkspace notifyScopeBorrowed(T workspace);

    void setWorkspaceName(T workspace, String name);

    String getWorkspaceName(T workspace);

    INDArray leverageTo(T toWorkspace, INDArray array);

    INDArray create(T workspace, int... shape);

    INDArray create(T workspace, int[] shape, char ordering);

    INDArray createUninitialized(T workspace, int... shape);

    INDArray createUninitialized(T workspace, int[] shape, char order);



}
