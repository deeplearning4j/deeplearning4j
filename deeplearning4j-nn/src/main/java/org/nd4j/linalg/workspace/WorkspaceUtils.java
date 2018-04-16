package org.nd4j.linalg.workspace;

import lombok.NonNull;
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

    public static void assertOpenAndActive(@NonNull String ws, @NonNull String errorMsg){
        if(!Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(ws)){
            throw new IllegalStateException(errorMsg);
        }
    }

    public static void assertOpenActiveAndCurrent(@NonNull String ws, @NonNull String errorMsg){
        if(!Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(ws)){
            throw new IllegalStateException(errorMsg + " - workspace is not open and active");
        }
        MemoryWorkspace currWs = Nd4j.getMemoryManager().getCurrentWorkspace();
        if(currWs == null || !ws.equals(currWs.getId())){
            throw new IllegalStateException(errorMsg + " - not the current workspace (current workspace: "
                    + (currWs == null ? null : currWs.getId()));
        }
    }

}
