package org.nd4j.jita.workspace;

import lombok.NonNull;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

/**
 * @author raver119@gmail.com
 */
public class CudaWorkspace extends Nd4jWorkspace {
    public CudaWorkspace(@NonNull WorkspaceConfiguration configuration) {
        super(configuration);
    }

    public CudaWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId) {
        super(configuration, workspaceId);
    }
}
