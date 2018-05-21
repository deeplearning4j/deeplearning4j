package org.nd4j.linalg.exception;

/**
 * ND4JIllegalStateException: thrown on invalid operations (for example, matrix multiplication with invalid arrays)
 *
 * @author Alex Black
 */
public class ND4JIllegalStateException extends ND4JException {
    public ND4JIllegalStateException() {}

    public ND4JIllegalStateException(String message) {
        super(message);
    }

    public ND4JIllegalStateException(String message, Throwable cause) {
        super(message, cause);
    }

    public ND4JIllegalStateException(Throwable cause) {
        super(cause);
    }
}
