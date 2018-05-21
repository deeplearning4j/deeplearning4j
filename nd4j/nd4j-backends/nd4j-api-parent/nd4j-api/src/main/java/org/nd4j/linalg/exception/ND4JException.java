package org.nd4j.linalg.exception;

/**
 * Base (unchecked) exception for ND4J errors
 *
 * @author Alex Black
 */
public class ND4JException extends RuntimeException {
    public ND4JException() {}

    public ND4JException(String message) {
        super(message);
    }

    public ND4JException(String message, Throwable cause) {
        super(message, cause);
    }

    public ND4JException(Throwable cause) {
        super(cause);
    }
}
