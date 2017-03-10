package org.deeplearning4j.exception;

/**
 * Base exception for DL4J
 *
 * @author Alex Black
 */
public class DL4JException extends RuntimeException {

    public DL4JException() {}

    public DL4JException(String message) {
        super(message);
    }

    public DL4JException(String message, Throwable cause) {
        super(message, cause);
    }

    public DL4JException(Throwable cause) {
        super(cause);
    }
}
