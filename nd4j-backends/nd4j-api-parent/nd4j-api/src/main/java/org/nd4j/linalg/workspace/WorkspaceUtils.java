package org.nd4j.linalg.workspace;

import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import java.util.ArrayList;
import java.util.List;

/**
 * Workspace utility methods
 *
 * @author Alex Black
 */
public class WorkspaceUtils {

    private WorkspaceUtils() {
    }

    /**
     * Assert that no workspaces are currently open
     *
     * @param msg Message to include in the exception, if required
     */
    public static void assertNoWorkspacesOpen(String msg) throws ND4JWorkspaceException {
        if (Nd4j.getWorkspaceManager().anyWorkspaceActiveForCurrentThread()) {
            List<MemoryWorkspace> l = Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread();
            List<String> workspaces = new ArrayList<>(l.size());
            for (MemoryWorkspace ws : l) {
                workspaces.add(ws.getId());
            }
            throw new ND4JWorkspaceException(msg + " - Open/active workspaces: " + workspaces);
        }
    }

    /**
     * Assert that the specified workspace is open and active
     *
     * @param ws       Name of the workspace to assert open and active
     * @param errorMsg Message to include in the exception, if required
     */
    public static void assertOpenAndActive(@NonNull String ws, @NonNull String errorMsg) throws ND4JWorkspaceException {
        if (!Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(ws)) {
            throw new ND4JWorkspaceException(errorMsg);
        }
    }

    /**
     * Assert that the specified workspace is open, active, and is the current workspace
     *
     * @param ws       Name of the workspace to assert open/active/current
     * @param errorMsg Message to include in the exception, if required
     */
    public static void assertOpenActiveAndCurrent(@NonNull String ws, @NonNull String errorMsg) throws ND4JWorkspaceException {
        if (!Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(ws)) {
            throw new ND4JWorkspaceException(errorMsg + " - workspace is not open and active");
        }
        MemoryWorkspace currWs = Nd4j.getMemoryManager().getCurrentWorkspace();
        if (currWs == null || !ws.equals(currWs.getId())) {
            throw new ND4JWorkspaceException(errorMsg + " - not the current workspace (current workspace: "
                    + (currWs == null ? null : currWs.getId()));
        }
    }

    /**
     * Assert that the specified array is valid, in terms of workspaces: i.e., if it is attached (and not in a circular
     * workspace), assert that the workspace is open, and that the data is not from an old generation.
     * @param array Array to check
     * @param msg   Message (prefix) to include in the exception, if required. May be null
     */
    public static void assertValidArray(INDArray array, String msg){
        if(array == null || !array.isAttached()){
            return;
        }

        val ws = array.data().getParentWorkspace();

        if (ws.getWorkspaceType() != MemoryWorkspace.Type.CIRCULAR) {

            if (!ws.isScopeActive()) {
                throw new ND4JWorkspaceException( (msg == null ? "" : msg + ": ") + "Array uses leaked workspace pointer " +
                        "from workspace " + ws.getId() + "\nAll open workspaces: " + allOpenWorkspaces());
            }

            if (ws.getGenerationId() != array.data().getGenerationId()) {
                throw new ND4JWorkspaceException( (msg == null ? "" : msg + ": ") + "Array outdated workspace pointer " +
                        "from workspace " + ws.getId() + " (array generation " + array.data().getGenerationId() +
                        ", current workspace generation " + ws.getGenerationId()  + ")\nAll open workspaces: " + allOpenWorkspaces());
            }
        }
    }

    private static List<String> allOpenWorkspaces(){
        List<MemoryWorkspace> l = Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread();
        List<String> workspaces = new ArrayList<>(l.size());
        for( MemoryWorkspace ws : l){
            if(ws.isScopeActive()) {
                workspaces.add(ws.getId());
            }
        }
        return workspaces;
    }
}
