package org.nd4j.linalg.exception;

/**
 * An unchecked (runtime) exception that specifies that the requested workspace does not exist
 *
 * @author Alex Black
 */
public class Nd4jNoSuchWorkspaceException extends RuntimeException {

    public Nd4jNoSuchWorkspaceException(String msg){
        super(msg);
    }

    public Nd4jNoSuchWorkspaceException(String msg, Throwable cause){
        super(msg, cause);
    }

}
