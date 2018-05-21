package org.nd4j.linalg.workspace;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

/**
 * An {@link AutoCloseable} for multiple workspaces
 * NOTE: Workspaces are closed in REVERSED order to how they are provided to the costructor;
 * this is to mirror nested workspace use
 *
 * @author Alex Black
 */
@Data
public class WorkspacesCloseable implements AutoCloseable {
    private MemoryWorkspace[] workspaces;

    public WorkspacesCloseable(@NonNull MemoryWorkspace... workspaces){
        this.workspaces = workspaces;
    }

    @Override
    public void close() {
        for( int i=workspaces.length-1; i>=0; i-- ){
            workspaces[i].close();
        }
    }
}
