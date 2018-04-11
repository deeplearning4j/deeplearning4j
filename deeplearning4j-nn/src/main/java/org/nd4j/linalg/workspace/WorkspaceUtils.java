package org.nd4j.linalg.workspace;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class WorkspaceUtils {

    private WorkspaceUtils() { }

    public static void assertNoWorkspacesOpen(String msg){
        if(Nd4j.getWorkspaceManager().anyWorkspaceActiveForCurrentThread()){
            List<MemoryWorkspace> l = Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread();
            List<String> workspaces = new ArrayList<>(l.size());
            for( MemoryWorkspace ws : l){
                workspaces.add(ws.getId());
            }
            throw new IllegalStateException(msg + " - Open/active workspaces: " + workspaces);
        }
    }

    public static void assertOpenAndActive(String ws, String errorMsg){
        if(!Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(ws)){
            throw new IllegalStateException(errorMsg);
        }
    }

}
