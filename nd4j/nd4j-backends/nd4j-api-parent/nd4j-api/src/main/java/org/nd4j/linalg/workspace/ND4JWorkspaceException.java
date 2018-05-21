package org.nd4j.linalg.workspace;

/**
 * An exception to specify than a workspace-related error has occurred
 *
 * @author Alex Black
 */
public class ND4JWorkspaceException extends RuntimeException {

    public ND4JWorkspaceException(){

    }

    public ND4JWorkspaceException(Throwable cause){
        super(cause);
    }

    public ND4JWorkspaceException(String msg){
        super(msg);
    }

    public ND4JWorkspaceException(String msg, Throwable cause){
        super(msg, cause);
    }

}
