package org.deeplearning4j.exception;

/**
 * Exception signifying that the specified configuration is invalid
 *
 * @author Alex Black
 */
public class DL4JInvalidConfigException extends DL4JException {
    public DL4JInvalidConfigException() {}

    public DL4JInvalidConfigException(String message) {
        super(message);
    }

    public DL4JInvalidConfigException(String message, Throwable cause) {
        super(message, cause);
    }

    public DL4JInvalidConfigException(Throwable cause) {
        super(cause);
    }
}
