package org.nd4j.linalg.exception;

/**
 * ND4JIllegalStateException: thrown on invalid arguments
 *
 * @author Alex Black
 */
public class ND4JIllegalArgumentException extends ND4JException {
    public ND4JIllegalArgumentException() {}

    public ND4JIllegalArgumentException(String message) {
        super(message);
    }

    public ND4JIllegalArgumentException(String message, Throwable cause) {
        super(message, cause);
    }

    public ND4JIllegalArgumentException(Throwable cause) {
        super(cause);
    }
}
