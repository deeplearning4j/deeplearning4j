package org.nd4j.linalg.workspace;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

@AllArgsConstructor
@Data
public class WorkspacesCloseable implements AutoCloseable {
    private MemoryWorkspace[] workspaces;

    @Override
    public void close() {
        for(MemoryWorkspace ws : workspaces){
            ws.close();
        }
    }
}
